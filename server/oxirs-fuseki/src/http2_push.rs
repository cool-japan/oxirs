//! HTTP/2 Server Push for SPARQL results
//!
//! This module implements a `ServerPushManager` that tracks client-subscribed SPARQL queries
//! and simulates HTTP/2 push promises for proactive result delivery. When a client subscribes
//! to a query, the manager tracks it and can generate push promises when new results are
//! available, reducing round-trip latency for frequently-polled queries.
//!
//! # Architecture
//!
//! ```text
//! Client ──subscribe──> ServerPushManager
//!                          │
//!                          ├── SubscriptionRegistry (query -> clients)
//!                          ├── PushPromiseGenerator (creates push frames)
//!                          └── ResultCache (deduplication)
//! ```
//!
//! # HTTP/2 Push Promise Flow
//!
//! 1. Client sends initial SPARQL query with `Push-Subscribe: true` header
//! 2. Server registers the subscription in the push manager
//! 3. When data changes, server generates push promises for subscribed queries
//! 4. Push promise frames are sent before the actual response
//! 5. Client can cancel subscriptions or let them expire via TTL

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Unique identifier for a client connection.
pub type ClientId = u64;

/// Unique identifier for a subscription.
pub type SubscriptionId = u64;

/// Represents the state of a push promise.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PushPromiseState {
    /// Promise has been created but not sent.
    Pending,
    /// Promise has been sent to the client.
    Sent,
    /// Promise was fulfilled (response delivered).
    Fulfilled,
    /// Promise was cancelled by client or server.
    Cancelled,
    /// Promise expired before delivery.
    Expired,
}

/// A push promise frame containing the promised resource.
#[derive(Debug, Clone)]
pub struct PushPromise {
    /// The subscription that triggered this promise.
    pub subscription_id: SubscriptionId,
    /// The client receiving this promise.
    pub client_id: ClientId,
    /// The SPARQL query being pushed.
    pub query: String,
    /// The promised URI path.
    pub promised_path: String,
    /// HTTP method (always GET for push promises).
    pub method: String,
    /// Content type of the promised response.
    pub content_type: String,
    /// The serialized query results (if available).
    pub result_payload: Option<Vec<u8>>,
    /// State of this push promise.
    pub state: PushPromiseState,
    /// When this promise was created.
    pub created_at: Instant,
    /// ETag for cache validation.
    pub etag: Option<String>,
}

impl PushPromise {
    /// Returns the size of the result payload in bytes.
    pub fn payload_size(&self) -> usize {
        self.result_payload.as_ref().map_or(0, |p| p.len())
    }

    /// Returns true if this promise has expired.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Returns true if this promise is still actionable.
    pub fn is_actionable(&self) -> bool {
        matches!(
            self.state,
            PushPromiseState::Pending | PushPromiseState::Sent
        )
    }
}

/// A subscription to a SPARQL query for push notifications.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique subscription ID.
    pub id: SubscriptionId,
    /// The subscribed client.
    pub client_id: ClientId,
    /// The SPARQL query being subscribed to.
    pub query: String,
    /// The normalized query fingerprint for deduplication.
    pub query_fingerprint: u64,
    /// When this subscription was created.
    pub created_at: Instant,
    /// Time-to-live for this subscription.
    pub ttl: Duration,
    /// Maximum number of push promises to send.
    pub max_pushes: Option<usize>,
    /// Number of push promises sent so far.
    pub push_count: usize,
    /// Whether the subscription is active.
    pub active: bool,
    /// Preferred content type for results.
    pub content_type: String,
    /// Minimum interval between pushes.
    pub min_interval: Duration,
    /// Last time a push was sent.
    pub last_push_at: Option<Instant>,
}

impl Subscription {
    /// Returns true if this subscription has expired.
    pub fn is_expired(&self) -> bool {
        self.created_at.elapsed() > self.ttl
    }

    /// Returns true if this subscription has reached its push limit.
    pub fn is_exhausted(&self) -> bool {
        if let Some(max) = self.max_pushes {
            self.push_count >= max
        } else {
            false
        }
    }

    /// Returns true if this subscription can accept a new push right now.
    pub fn can_push(&self) -> bool {
        if !self.active || self.is_expired() || self.is_exhausted() {
            return false;
        }
        if let Some(last) = self.last_push_at {
            last.elapsed() >= self.min_interval
        } else {
            true
        }
    }

    /// Returns the remaining TTL.
    pub fn remaining_ttl(&self) -> Duration {
        self.ttl.saturating_sub(self.created_at.elapsed())
    }
}

/// Configuration for the server push manager.
#[derive(Debug, Clone)]
pub struct ServerPushConfig {
    /// Default TTL for subscriptions.
    pub default_ttl: Duration,
    /// Maximum subscriptions per client.
    pub max_subscriptions_per_client: usize,
    /// Maximum total subscriptions.
    pub max_total_subscriptions: usize,
    /// Maximum payload size for push promises (bytes).
    pub max_payload_size: usize,
    /// Default content type for push results.
    pub default_content_type: String,
    /// Minimum interval between pushes for a subscription.
    pub min_push_interval: Duration,
    /// Whether to enable ETag-based deduplication.
    pub enable_etag_dedup: bool,
    /// TTL for push promises before expiration.
    pub promise_ttl: Duration,
    /// Maximum pending promises per client.
    pub max_pending_per_client: usize,
}

impl Default for ServerPushConfig {
    fn default() -> Self {
        Self {
            default_ttl: Duration::from_secs(300), // 5 minutes
            max_subscriptions_per_client: 10,
            max_total_subscriptions: 1000,
            max_payload_size: 1024 * 1024, // 1 MB
            default_content_type: "application/sparql-results+json".to_string(),
            min_push_interval: Duration::from_secs(1),
            enable_etag_dedup: true,
            promise_ttl: Duration::from_secs(30),
            max_pending_per_client: 50,
        }
    }
}

/// Statistics about the push manager's operation.
#[derive(Debug, Clone, Default)]
pub struct PushStats {
    /// Total subscriptions created.
    pub total_subscriptions_created: u64,
    /// Currently active subscriptions.
    pub active_subscriptions: usize,
    /// Total push promises generated.
    pub total_promises_generated: u64,
    /// Total push promises fulfilled.
    pub total_promises_fulfilled: u64,
    /// Total push promises cancelled.
    pub total_promises_cancelled: u64,
    /// Total push promises expired.
    pub total_promises_expired: u64,
    /// Total bytes pushed.
    pub total_bytes_pushed: u64,
    /// Number of deduplication hits (skipped pushes due to same ETag).
    pub dedup_hits: u64,
    /// Number of rate-limited pushes.
    pub rate_limited: u64,
    /// Number of expired subscriptions cleaned up.
    pub expired_cleanups: u64,
}

/// The server push manager tracks subscriptions and generates push promises.
pub struct ServerPushManager {
    config: ServerPushConfig,
    subscriptions: HashMap<SubscriptionId, Subscription>,
    /// Map from client ID to their subscription IDs.
    client_subscriptions: HashMap<ClientId, HashSet<SubscriptionId>>,
    /// Map from query fingerprint to subscription IDs (for efficient notification).
    query_index: HashMap<u64, HashSet<SubscriptionId>>,
    /// Pending push promises.
    pending_promises: Vec<PushPromise>,
    /// Last known ETags per (subscription_id) for dedup.
    last_etags: HashMap<SubscriptionId, String>,
    /// Next subscription ID.
    next_sub_id: SubscriptionId,
    /// Statistics.
    stats: PushStats,
}

impl ServerPushManager {
    /// Creates a new server push manager with the given configuration.
    pub fn new(config: ServerPushConfig) -> Self {
        Self {
            config,
            subscriptions: HashMap::new(),
            client_subscriptions: HashMap::new(),
            query_index: HashMap::new(),
            pending_promises: Vec::new(),
            last_etags: HashMap::new(),
            next_sub_id: 1,
            stats: PushStats::default(),
        }
    }

    /// Creates a new server push manager with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(ServerPushConfig::default())
    }

    /// Returns the current configuration.
    pub fn config(&self) -> &ServerPushConfig {
        &self.config
    }

    /// Subscribe a client to a SPARQL query.
    ///
    /// Returns the subscription ID on success, or an error description.
    pub fn subscribe(
        &mut self,
        client_id: ClientId,
        query: String,
        ttl: Option<Duration>,
        max_pushes: Option<usize>,
    ) -> Result<SubscriptionId, String> {
        // Check total limit
        if self.subscriptions.len() >= self.config.max_total_subscriptions {
            return Err("Maximum total subscriptions reached".to_string());
        }

        // Check per-client limit
        let client_subs = self.client_subscriptions.entry(client_id).or_default();
        if client_subs.len() >= self.config.max_subscriptions_per_client {
            return Err(format!(
                "Client {client_id} has reached the maximum of {} subscriptions",
                self.config.max_subscriptions_per_client
            ));
        }

        let sub_id = self.next_sub_id;
        self.next_sub_id += 1;

        let fingerprint = compute_query_fingerprint(&query);

        let subscription = Subscription {
            id: sub_id,
            client_id,
            query,
            query_fingerprint: fingerprint,
            created_at: Instant::now(),
            ttl: ttl.unwrap_or(self.config.default_ttl),
            max_pushes,
            push_count: 0,
            active: true,
            content_type: self.config.default_content_type.clone(),
            min_interval: self.config.min_push_interval,
            last_push_at: None,
        };

        self.subscriptions.insert(sub_id, subscription);
        client_subs.insert(sub_id);
        self.query_index
            .entry(fingerprint)
            .or_default()
            .insert(sub_id);

        self.stats.total_subscriptions_created += 1;
        self.stats.active_subscriptions = self.subscriptions.len();

        Ok(sub_id)
    }

    /// Unsubscribe a specific subscription.
    pub fn unsubscribe(&mut self, subscription_id: SubscriptionId) -> bool {
        if let Some(sub) = self.subscriptions.remove(&subscription_id) {
            // Remove from client index
            if let Some(client_subs) = self.client_subscriptions.get_mut(&sub.client_id) {
                client_subs.remove(&subscription_id);
                if client_subs.is_empty() {
                    self.client_subscriptions.remove(&sub.client_id);
                }
            }

            // Remove from query index
            if let Some(query_subs) = self.query_index.get_mut(&sub.query_fingerprint) {
                query_subs.remove(&subscription_id);
                if query_subs.is_empty() {
                    self.query_index.remove(&sub.query_fingerprint);
                }
            }

            self.last_etags.remove(&subscription_id);
            self.stats.active_subscriptions = self.subscriptions.len();
            true
        } else {
            false
        }
    }

    /// Unsubscribe all subscriptions for a client.
    pub fn unsubscribe_client(&mut self, client_id: ClientId) -> usize {
        let sub_ids: Vec<SubscriptionId> = self
            .client_subscriptions
            .get(&client_id)
            .map_or_else(Vec::new, |subs| subs.iter().copied().collect());

        let count = sub_ids.len();
        for sub_id in sub_ids {
            self.unsubscribe(sub_id);
        }
        count
    }

    /// Get a subscription by ID.
    pub fn get_subscription(&self, subscription_id: SubscriptionId) -> Option<&Subscription> {
        self.subscriptions.get(&subscription_id)
    }

    /// Get all subscription IDs for a client.
    pub fn client_subscription_ids(&self, client_id: ClientId) -> Vec<SubscriptionId> {
        self.client_subscriptions
            .get(&client_id)
            .map_or_else(Vec::new, |subs| subs.iter().copied().collect())
    }

    /// Returns the total number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Returns the number of unique clients with subscriptions.
    pub fn client_count(&self) -> usize {
        self.client_subscriptions.len()
    }

    /// Notify that results are available for a query, generating push promises.
    ///
    /// Returns the list of generated push promises.
    pub fn notify_results(
        &mut self,
        query: &str,
        result_payload: Vec<u8>,
        etag: Option<String>,
    ) -> Vec<PushPromise> {
        let fingerprint = compute_query_fingerprint(query);
        let mut promises = Vec::new();

        let sub_ids: Vec<SubscriptionId> = self
            .query_index
            .get(&fingerprint)
            .map_or_else(Vec::new, |subs| subs.iter().copied().collect());

        for sub_id in sub_ids {
            let can_push = self
                .subscriptions
                .get(&sub_id)
                .is_some_and(|sub| sub.can_push());

            if !can_push {
                if self
                    .subscriptions
                    .get(&sub_id)
                    .is_some_and(|s| s.active && !s.can_push())
                {
                    self.stats.rate_limited += 1;
                }
                continue;
            }

            // Check payload size
            if result_payload.len() > self.config.max_payload_size {
                continue;
            }

            // ETag deduplication
            if self.config.enable_etag_dedup {
                if let Some(ref new_etag) = etag {
                    if let Some(last_etag) = self.last_etags.get(&sub_id) {
                        if last_etag == new_etag {
                            self.stats.dedup_hits += 1;
                            continue;
                        }
                    }
                    self.last_etags.insert(sub_id, new_etag.clone());
                }
            }

            // Check pending promises limit
            let client_id = self
                .subscriptions
                .get(&sub_id)
                .map(|s| s.client_id)
                .unwrap_or(0);
            let pending_count = self
                .pending_promises
                .iter()
                .filter(|p| p.client_id == client_id && p.is_actionable())
                .count();
            if pending_count >= self.config.max_pending_per_client {
                continue;
            }

            if let Some(sub) = self.subscriptions.get_mut(&sub_id) {
                let promise = PushPromise {
                    subscription_id: sub_id,
                    client_id: sub.client_id,
                    query: sub.query.clone(),
                    promised_path: format!("/sparql/push/{sub_id}"),
                    method: "GET".to_string(),
                    content_type: sub.content_type.clone(),
                    result_payload: Some(result_payload.clone()),
                    state: PushPromiseState::Pending,
                    created_at: Instant::now(),
                    etag: etag.clone(),
                };

                sub.push_count += 1;
                sub.last_push_at = Some(Instant::now());

                self.stats.total_promises_generated += 1;
                self.stats.total_bytes_pushed += result_payload.len() as u64;

                promises.push(promise.clone());
                self.pending_promises.push(promise);
            }
        }

        promises
    }

    /// Mark a push promise as fulfilled.
    pub fn fulfill_promise(&mut self, subscription_id: SubscriptionId) -> bool {
        let mut found = false;
        for promise in &mut self.pending_promises {
            if promise.subscription_id == subscription_id
                && promise.state == PushPromiseState::Pending
            {
                promise.state = PushPromiseState::Fulfilled;
                self.stats.total_promises_fulfilled += 1;
                found = true;
                break;
            }
        }
        found
    }

    /// Cancel a pending push promise.
    pub fn cancel_promise(&mut self, subscription_id: SubscriptionId) -> bool {
        let mut found = false;
        for promise in &mut self.pending_promises {
            if promise.subscription_id == subscription_id
                && promise.state == PushPromiseState::Pending
            {
                promise.state = PushPromiseState::Cancelled;
                self.stats.total_promises_cancelled += 1;
                found = true;
                break;
            }
        }
        found
    }

    /// Clean up expired subscriptions and promises.
    pub fn cleanup_expired(&mut self) -> usize {
        let expired_sub_ids: Vec<SubscriptionId> = self
            .subscriptions
            .iter()
            .filter(|(_, sub)| sub.is_expired() || sub.is_exhausted())
            .map(|(id, _)| *id)
            .collect();

        let count = expired_sub_ids.len();
        for sub_id in expired_sub_ids {
            self.unsubscribe(sub_id);
        }

        // Clean up expired promises
        let promise_ttl = self.config.promise_ttl;
        let mut expired_promise_count = 0u64;
        for promise in &mut self.pending_promises {
            if promise.is_expired(promise_ttl) && promise.state == PushPromiseState::Pending {
                promise.state = PushPromiseState::Expired;
                expired_promise_count += 1;
            }
        }
        self.stats.total_promises_expired += expired_promise_count;

        // Remove non-actionable promises
        self.pending_promises.retain(|p| p.is_actionable());

        self.stats.expired_cleanups += count as u64;
        count
    }

    /// Returns the pending promises for a client.
    pub fn pending_promises_for_client(&self, client_id: ClientId) -> Vec<&PushPromise> {
        self.pending_promises
            .iter()
            .filter(|p| p.client_id == client_id && p.is_actionable())
            .collect()
    }

    /// Returns current statistics.
    pub fn stats(&self) -> &PushStats {
        &self.stats
    }

    /// Returns a snapshot of the current statistics (cloned).
    pub fn stats_snapshot(&self) -> PushStats {
        let mut stats = self.stats.clone();
        stats.active_subscriptions = self.subscriptions.len();
        stats
    }

    /// Checks if a client has any active subscriptions.
    pub fn has_subscriptions(&self, client_id: ClientId) -> bool {
        self.client_subscriptions
            .get(&client_id)
            .is_some_and(|subs| !subs.is_empty())
    }

    /// Returns all unique query fingerprints with active subscriptions.
    pub fn active_query_fingerprints(&self) -> Vec<u64> {
        self.query_index.keys().copied().collect()
    }

    /// Returns the number of subscriptions for a given query.
    pub fn query_subscriber_count(&self, query: &str) -> usize {
        let fingerprint = compute_query_fingerprint(query);
        self.query_index
            .get(&fingerprint)
            .map_or(0, |subs| subs.len())
    }
}

/// Computes a simple fingerprint for a SPARQL query.
///
/// This normalizes whitespace and produces a hash for lookup.
fn compute_query_fingerprint(query: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Normalize: lowercase, collapse whitespace
    let normalized: String = query
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
        .to_lowercase();

    let mut hasher = DefaultHasher::new();
    normalized.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_manager() -> ServerPushManager {
        ServerPushManager::with_defaults()
    }

    fn small_manager() -> ServerPushManager {
        ServerPushManager::new(ServerPushConfig {
            max_subscriptions_per_client: 3,
            max_total_subscriptions: 10,
            max_payload_size: 1024,
            default_ttl: Duration::from_secs(60),
            min_push_interval: Duration::from_millis(0), // no throttle for tests
            promise_ttl: Duration::from_secs(5),
            max_pending_per_client: 5,
            ..ServerPushConfig::default()
        })
    }

    // ── Subscription creation ───────────────────────────────────────────────

    #[test]
    fn test_subscribe_basic() {
        let mut mgr = default_manager();
        let sub_id = mgr
            .subscribe(1, "SELECT * WHERE { ?s ?p ?o }".to_string(), None, None)
            .expect("subscribe should succeed");
        assert!(sub_id > 0);
        assert_eq!(mgr.subscription_count(), 1);
        assert_eq!(mgr.client_count(), 1);
    }

    #[test]
    fn test_subscribe_multiple_clients() {
        let mut mgr = default_manager();
        mgr.subscribe(1, "SELECT ?s WHERE { ?s ?p ?o }".to_string(), None, None)
            .expect("ok");
        mgr.subscribe(2, "SELECT ?s WHERE { ?s ?p ?o }".to_string(), None, None)
            .expect("ok");
        assert_eq!(mgr.subscription_count(), 2);
        assert_eq!(mgr.client_count(), 2);
    }

    #[test]
    fn test_subscribe_per_client_limit() {
        let mut mgr = small_manager();
        for i in 0..3 {
            mgr.subscribe(1, format!("SELECT * WHERE {{ ?s ?p {i} }}"), None, None)
                .expect("ok");
        }
        let result = mgr.subscribe(1, "SELECT * WHERE { ?s ?p 3 }".to_string(), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_subscribe_total_limit() {
        let mut mgr = ServerPushManager::new(ServerPushConfig {
            max_total_subscriptions: 2,
            max_subscriptions_per_client: 100,
            ..ServerPushConfig::default()
        });
        mgr.subscribe(1, "q1".to_string(), None, None).expect("ok");
        mgr.subscribe(2, "q2".to_string(), None, None).expect("ok");
        let result = mgr.subscribe(3, "q3".to_string(), None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_subscribe_with_custom_ttl() {
        let mut mgr = default_manager();
        let sub_id = mgr
            .subscribe(1, "q".to_string(), Some(Duration::from_secs(10)), None)
            .expect("ok");
        let sub = mgr.get_subscription(sub_id).expect("exists");
        assert_eq!(sub.ttl, Duration::from_secs(10));
    }

    #[test]
    fn test_subscribe_with_max_pushes() {
        let mut mgr = default_manager();
        let sub_id = mgr
            .subscribe(1, "q".to_string(), None, Some(5))
            .expect("ok");
        let sub = mgr.get_subscription(sub_id).expect("exists");
        assert_eq!(sub.max_pushes, Some(5));
    }

    // ── Unsubscribe ─────────────────────────────────────────────────────────

    #[test]
    fn test_unsubscribe() {
        let mut mgr = default_manager();
        let sub_id = mgr.subscribe(1, "q".to_string(), None, None).expect("ok");
        assert!(mgr.unsubscribe(sub_id));
        assert_eq!(mgr.subscription_count(), 0);
    }

    #[test]
    fn test_unsubscribe_nonexistent() {
        let mut mgr = default_manager();
        assert!(!mgr.unsubscribe(999));
    }

    #[test]
    fn test_unsubscribe_client() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "q1".to_string(), None, None).expect("ok");
        mgr.subscribe(1, "q2".to_string(), None, None).expect("ok");
        mgr.subscribe(2, "q3".to_string(), None, None).expect("ok");
        let removed = mgr.unsubscribe_client(1);
        assert_eq!(removed, 2);
        assert_eq!(mgr.subscription_count(), 1);
    }

    // ── Notification and push promises ──────────────────────────────────────

    #[test]
    fn test_notify_generates_promises() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "SELECT * WHERE { ?s ?p ?o }".to_string(), None, None)
            .expect("ok");
        let promises =
            mgr.notify_results("SELECT * WHERE { ?s ?p ?o }", b"result-data".to_vec(), None);
        assert_eq!(promises.len(), 1);
        assert_eq!(promises[0].state, PushPromiseState::Pending);
    }

    #[test]
    fn test_notify_multiple_subscribers() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "SELECT ?x WHERE { ?x a :Thing }".to_string(), None, None)
            .expect("ok");
        mgr.subscribe(2, "SELECT ?x WHERE { ?x a :Thing }".to_string(), None, None)
            .expect("ok");
        let promises =
            mgr.notify_results("SELECT ?x WHERE { ?x a :Thing }", b"data".to_vec(), None);
        assert_eq!(promises.len(), 2);
    }

    #[test]
    fn test_notify_no_match() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "SELECT * WHERE { ?s ?p ?o }".to_string(), None, None)
            .expect("ok");
        let promises = mgr.notify_results("SELECT * WHERE { ?x ?y ?z }", b"data".to_vec(), None);
        assert_eq!(promises.len(), 0);
    }

    #[test]
    fn test_notify_payload_too_large() {
        let mut mgr = small_manager(); // max 1024 bytes
        mgr.subscribe(1, "q".to_string(), None, None).expect("ok");
        let big_payload = vec![0u8; 2048];
        let promises = mgr.notify_results("q", big_payload, None);
        assert_eq!(promises.len(), 0);
    }

    #[test]
    fn test_etag_deduplication() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "q".to_string(), None, None).expect("ok");

        let p1 = mgr.notify_results("q", b"data1".to_vec(), Some("etag1".to_string()));
        assert_eq!(p1.len(), 1);

        // Same etag: should be deduplicated
        let p2 = mgr.notify_results("q", b"data1".to_vec(), Some("etag1".to_string()));
        assert_eq!(p2.len(), 0);
        assert_eq!(mgr.stats().dedup_hits, 1);

        // Different etag: should push
        let p3 = mgr.notify_results("q", b"data2".to_vec(), Some("etag2".to_string()));
        assert_eq!(p3.len(), 1);
    }

    // ── Promise lifecycle ───────────────────────────────────────────────────

    #[test]
    fn test_fulfill_promise() {
        let mut mgr = small_manager();
        let sub_id = mgr.subscribe(1, "q".to_string(), None, None).expect("ok");
        mgr.notify_results("q", b"data".to_vec(), None);
        assert!(mgr.fulfill_promise(sub_id));
        assert_eq!(mgr.stats().total_promises_fulfilled, 1);
    }

    #[test]
    fn test_cancel_promise() {
        let mut mgr = small_manager();
        let sub_id = mgr.subscribe(1, "q".to_string(), None, None).expect("ok");
        mgr.notify_results("q", b"data".to_vec(), None);
        assert!(mgr.cancel_promise(sub_id));
        assert_eq!(mgr.stats().total_promises_cancelled, 1);
    }

    #[test]
    fn test_pending_promises_for_client() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "q1".to_string(), None, None).expect("ok");
        mgr.subscribe(2, "q2".to_string(), None, None).expect("ok");
        mgr.notify_results("q1", b"data".to_vec(), None);
        mgr.notify_results("q2", b"data".to_vec(), None);

        let client1_pending = mgr.pending_promises_for_client(1);
        assert_eq!(client1_pending.len(), 1);
        let client2_pending = mgr.pending_promises_for_client(2);
        assert_eq!(client2_pending.len(), 1);
    }

    // ── Expiration and cleanup ──────────────────────────────────────────────

    #[test]
    fn test_subscription_expiry_detection() {
        let sub = Subscription {
            id: 1,
            client_id: 1,
            query: "q".to_string(),
            query_fingerprint: 0,
            created_at: Instant::now() - Duration::from_secs(120),
            ttl: Duration::from_secs(60),
            max_pushes: None,
            push_count: 0,
            active: true,
            content_type: "application/json".to_string(),
            min_interval: Duration::from_millis(0),
            last_push_at: None,
        };
        assert!(sub.is_expired());
        assert!(!sub.can_push());
    }

    #[test]
    fn test_subscription_exhaustion() {
        let sub = Subscription {
            id: 1,
            client_id: 1,
            query: "q".to_string(),
            query_fingerprint: 0,
            created_at: Instant::now(),
            ttl: Duration::from_secs(300),
            max_pushes: Some(5),
            push_count: 5,
            active: true,
            content_type: "application/json".to_string(),
            min_interval: Duration::from_millis(0),
            last_push_at: None,
        };
        assert!(sub.is_exhausted());
        assert!(!sub.can_push());
    }

    #[test]
    fn test_cleanup_expired() {
        let mut mgr = ServerPushManager::new(ServerPushConfig {
            default_ttl: Duration::from_millis(1),
            min_push_interval: Duration::from_millis(0),
            promise_ttl: Duration::from_millis(1),
            ..ServerPushConfig::default()
        });
        mgr.subscribe(1, "q".to_string(), Some(Duration::from_millis(1)), None)
            .expect("ok");

        // Wait for expiration
        std::thread::sleep(Duration::from_millis(10));

        let cleaned = mgr.cleanup_expired();
        assert_eq!(cleaned, 1);
        assert_eq!(mgr.subscription_count(), 0);
    }

    // ── Query fingerprinting ────────────────────────────────────────────────

    #[test]
    fn test_query_fingerprint_normalization() {
        let fp1 = compute_query_fingerprint("SELECT * WHERE { ?s ?p ?o }");
        let fp2 = compute_query_fingerprint("SELECT  *  WHERE  {  ?s  ?p  ?o  }");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_query_fingerprint_case_insensitive() {
        let fp1 = compute_query_fingerprint("SELECT * WHERE { ?s ?p ?o }");
        let fp2 = compute_query_fingerprint("select * where { ?s ?p ?o }");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_query_fingerprint_different_queries() {
        let fp1 = compute_query_fingerprint("SELECT ?s WHERE { ?s ?p ?o }");
        let fp2 = compute_query_fingerprint("SELECT ?p WHERE { ?s ?p ?o }");
        assert_ne!(fp1, fp2);
    }

    // ── Stats ───────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_tracking() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "q".to_string(), None, None).expect("ok");
        mgr.notify_results("q", b"data".to_vec(), None);
        let stats = mgr.stats_snapshot();
        assert_eq!(stats.total_subscriptions_created, 1);
        assert_eq!(stats.active_subscriptions, 1);
        assert_eq!(stats.total_promises_generated, 1);
        assert!(stats.total_bytes_pushed > 0);
    }

    #[test]
    fn test_stats_after_unsubscribe() {
        let mut mgr = small_manager();
        let sub_id = mgr.subscribe(1, "q".to_string(), None, None).expect("ok");
        mgr.unsubscribe(sub_id);
        let stats = mgr.stats_snapshot();
        assert_eq!(stats.total_subscriptions_created, 1);
        assert_eq!(stats.active_subscriptions, 0);
    }

    // ── Helper queries ──────────────────────────────────────────────────────

    #[test]
    fn test_has_subscriptions() {
        let mut mgr = default_manager();
        assert!(!mgr.has_subscriptions(1));
        mgr.subscribe(1, "q".to_string(), None, None).expect("ok");
        assert!(mgr.has_subscriptions(1));
    }

    #[test]
    fn test_client_subscription_ids() {
        let mut mgr = small_manager();
        let s1 = mgr.subscribe(1, "q1".to_string(), None, None).expect("ok");
        let s2 = mgr.subscribe(1, "q2".to_string(), None, None).expect("ok");
        let ids = mgr.client_subscription_ids(1);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&s1));
        assert!(ids.contains(&s2));
    }

    #[test]
    fn test_active_query_fingerprints() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "q1".to_string(), None, None).expect("ok");
        mgr.subscribe(2, "q2".to_string(), None, None).expect("ok");
        let fps = mgr.active_query_fingerprints();
        assert_eq!(fps.len(), 2);
    }

    #[test]
    fn test_query_subscriber_count() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "SELECT * WHERE { ?s ?p ?o }".to_string(), None, None)
            .expect("ok");
        mgr.subscribe(2, "SELECT * WHERE { ?s ?p ?o }".to_string(), None, None)
            .expect("ok");
        assert_eq!(mgr.query_subscriber_count("SELECT * WHERE { ?s ?p ?o }"), 2);
    }

    // ── Push promise properties ─────────────────────────────────────────────

    #[test]
    fn test_push_promise_payload_size() {
        let promise = PushPromise {
            subscription_id: 1,
            client_id: 1,
            query: "q".to_string(),
            promised_path: "/push/1".to_string(),
            method: "GET".to_string(),
            content_type: "application/json".to_string(),
            result_payload: Some(vec![1, 2, 3, 4, 5]),
            state: PushPromiseState::Pending,
            created_at: Instant::now(),
            etag: None,
        };
        assert_eq!(promise.payload_size(), 5);
    }

    #[test]
    fn test_push_promise_no_payload() {
        let promise = PushPromise {
            subscription_id: 1,
            client_id: 1,
            query: "q".to_string(),
            promised_path: "/push/1".to_string(),
            method: "GET".to_string(),
            content_type: "application/json".to_string(),
            result_payload: None,
            state: PushPromiseState::Pending,
            created_at: Instant::now(),
            etag: None,
        };
        assert_eq!(promise.payload_size(), 0);
    }

    #[test]
    fn test_push_promise_actionable_states() {
        for state in [PushPromiseState::Pending, PushPromiseState::Sent] {
            let promise = PushPromise {
                subscription_id: 1,
                client_id: 1,
                query: "q".to_string(),
                promised_path: "/push/1".to_string(),
                method: "GET".to_string(),
                content_type: "application/json".to_string(),
                result_payload: None,
                state,
                created_at: Instant::now(),
                etag: None,
            };
            assert!(promise.is_actionable());
        }
        for state in [
            PushPromiseState::Fulfilled,
            PushPromiseState::Cancelled,
            PushPromiseState::Expired,
        ] {
            let promise = PushPromise {
                subscription_id: 1,
                client_id: 1,
                query: "q".to_string(),
                promised_path: "/push/1".to_string(),
                method: "GET".to_string(),
                content_type: "application/json".to_string(),
                result_payload: None,
                state,
                created_at: Instant::now(),
                etag: None,
            };
            assert!(!promise.is_actionable());
        }
    }

    // ── Remaining TTL ───────────────────────────────────────────────────────

    #[test]
    fn test_remaining_ttl() {
        let sub = Subscription {
            id: 1,
            client_id: 1,
            query: "q".to_string(),
            query_fingerprint: 0,
            created_at: Instant::now(),
            ttl: Duration::from_secs(300),
            max_pushes: None,
            push_count: 0,
            active: true,
            content_type: "application/json".to_string(),
            min_interval: Duration::from_millis(0),
            last_push_at: None,
        };
        let remaining = sub.remaining_ttl();
        assert!(remaining <= Duration::from_secs(300));
        assert!(remaining > Duration::from_secs(290));
    }

    // ── Max pushes exhaustion via notify ─────────────────────────────────

    #[test]
    fn test_max_pushes_exhaustion() {
        let mut mgr = small_manager();
        mgr.subscribe(1, "q".to_string(), None, Some(2))
            .expect("ok");

        let p1 = mgr.notify_results("q", b"d1".to_vec(), None);
        assert_eq!(p1.len(), 1);

        let p2 = mgr.notify_results("q", b"d2".to_vec(), None);
        assert_eq!(p2.len(), 1);

        // Third push should be blocked (exhausted)
        let p3 = mgr.notify_results("q", b"d3".to_vec(), None);
        assert_eq!(p3.len(), 0);
    }

    // ── Config ──────────────────────────────────────────────────────────────

    #[test]
    fn test_config_access() {
        let mgr = default_manager();
        let cfg = mgr.config();
        assert_eq!(cfg.default_ttl, Duration::from_secs(300));
        assert!(cfg.enable_etag_dedup);
    }

    #[test]
    fn test_default_config() {
        let cfg = ServerPushConfig::default();
        assert_eq!(cfg.max_subscriptions_per_client, 10);
        assert_eq!(cfg.max_total_subscriptions, 1000);
        assert_eq!(cfg.max_payload_size, 1024 * 1024);
    }
}
