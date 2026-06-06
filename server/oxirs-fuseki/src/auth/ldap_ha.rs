//! LDAP High Availability (HA) failover pool for OxiRS Fuseki
//!
//! Provides a multi-server LDAP pool with:
//! - Circuit breaker per server (auto-open after threshold failures, auto-reset after timeout)
//! - Round-robin and primary-preferred read policies
//! - Automatic failover for bind and search operations
//! - Mock transport for testing (no external LDAP crate required)

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ─── Public error type ────────────────────────────────────────────────────────

/// All errors that can be returned by the HA pool.
#[derive(Debug, thiserror::Error)]
pub enum LdapHaError {
    /// No server is currently healthy (all circuit breakers open).
    #[error("No available LDAP server")]
    NoServersAvailable,

    /// Every server was tried and all failed.
    #[error("All servers unreachable: {servers:?}")]
    AllServersUnreachable { servers: Vec<String> },

    /// Bind/authenticate failed on all available servers.
    #[error("Authentication failed: {reason}")]
    AuthFailed { reason: String },

    /// Search failed on all available servers.
    #[error("Search failed: {reason}")]
    SearchFailed { reason: String },
}

/// Convenience alias.
pub type LdapResult<T> = Result<T, LdapHaError>;

// ─── Domain types ─────────────────────────────────────────────────────────────

/// Role of a server in the HA pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdapServerRole {
    /// The writable primary — all bind/write operations target this server.
    Primary,
    /// A read-only replica — used for search/read operations under most policies.
    Replica,
}

/// Configuration for a single LDAP server in the pool.
pub struct LdapServer {
    /// LDAP URI, e.g. `ldap://primary.example.com:389`.
    pub uri: String,
    /// Whether this server is the primary or a replica.
    pub role: LdapServerRole,
    /// Lower value = higher priority (primary should be 0, replicas 1, 2, …).
    pub priority: u8,
}

/// Coarse health classification for a server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LdapServerHealth {
    /// Circuit breaker closed; server is responding normally.
    Healthy,
    /// Some recent failures but still below threshold.
    Degraded,
    /// Circuit breaker open; server is considered unreachable.
    Unreachable,
}

// ─── Circuit breaker ──────────────────────────────────────────────────────────

/// Per-server circuit breaker state.
pub struct LdapCircuitBreaker {
    /// Consecutive failures since last success (or since last reset).
    failure_count: u32,
    /// Time of the most recent failure (used for reset timeout logic).
    last_failure: Option<Instant>,
    /// Number of consecutive failures that opens the circuit.
    threshold: u32,
    /// How long after opening before the circuit is reset (half-open attempt).
    reset_timeout: Duration,
}

impl LdapCircuitBreaker {
    /// Create a new circuit breaker with the given thresholds.
    pub fn new(threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_count: 0,
            last_failure: None,
            threshold,
            reset_timeout,
        }
    }

    /// Returns `true` when the circuit is open **and** the reset timeout has
    /// not yet elapsed (i.e. the server should be skipped).
    pub fn is_open(&self) -> bool {
        if self.failure_count < self.threshold {
            return false;
        }
        // Circuit tripped — check if reset_timeout has elapsed.
        match self.last_failure {
            None => false,
            Some(t) => t.elapsed() < self.reset_timeout,
        }
    }

    /// Record a successful operation — resets the counter.
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.last_failure = None;
    }

    /// Record a failed operation — increments the counter and timestamps.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure = Some(Instant::now());
    }

    /// Coarse health classification.
    pub fn health(&self) -> LdapServerHealth {
        if self.is_open() {
            LdapServerHealth::Unreachable
        } else if self.failure_count > 0 {
            LdapServerHealth::Degraded
        } else {
            LdapServerHealth::Healthy
        }
    }
}

// ─── Read policy ─────────────────────────────────────────────────────────────

/// Policy controlling which server is selected for read/search operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadPolicy {
    /// Rotate through healthy servers in priority order (replicas first when
    /// available, fallback to primary).
    RoundRobin,
    /// Prefer the primary for reads; fall back to replicas only if the primary
    /// circuit breaker is open.
    PrimaryPreferred,
}

// ─── Session / entry stubs (no real LDAP crate) ───────────────────────────────

/// A successfully bound LDAP session (stub — real implementations would hold a
/// live connection; here it carries only the metadata needed by callers).
#[derive(Debug)]
pub struct LdapBoundSession {
    /// URI of the server that accepted the bind.
    pub server_uri: String,
    /// Username (DN or simple name) that was bound.
    pub username: String,
}

/// A single entry returned from an LDAP search.
#[derive(Debug)]
pub struct LdapEntry {
    /// Distinguished Name of the entry.
    pub dn: String,
    /// Attribute name → list of values.
    pub attrs: HashMap<String, Vec<String>>,
}

// ─── Mock transport ───────────────────────────────────────────────────────────

type LdapMockAuthFn = Arc<dyn Fn(&str, &str, &str) -> bool + Send + Sync>;

/// Internal transport abstraction so that the HA pool can be tested without a
/// real LDAP server.
enum Transport {
    /// Production path — would call a real LDAP client library.
    Real,
    /// Test path — a closure that decides whether a (server_uri, username,
    /// password) triple is authenticated.
    Mock(LdapMockAuthFn),
}

// ─── HA pool ─────────────────────────────────────────────────────────────────

/// Entry stored in the pool: the server config plus its circuit breaker.
struct PoolEntry {
    server: LdapServer,
    breaker: Mutex<LdapCircuitBreaker>,
}

/// High-availability LDAP server pool.
///
/// # Thread safety
/// All mutable state is protected by `Mutex` so that the pool can be wrapped
/// in an `Arc` and shared across async tasks.
pub struct LdapHaPool {
    /// Sorted by `server.priority` (ascending = higher priority first).
    entries: Vec<PoolEntry>,
    /// Read / search routing policy.
    read_policy: ReadPolicy,
    /// Round-robin cursor (only used when policy is `RoundRobin`).
    rr_cursor: AtomicUsize,
    /// Transport (real or mock).
    transport: Transport,
}

impl LdapHaPool {
    // ── Constructors ─────────────────────────────────────────────────────────

    /// Create a pool with real (stubbed) transport.
    ///
    /// `servers` need not be sorted beforehand — the pool will sort them by
    /// `priority` internally.
    pub fn new(mut servers: Vec<LdapServer>) -> Self {
        servers.sort_by_key(|s| s.priority);
        let entries = servers
            .into_iter()
            .map(|s| PoolEntry {
                server: s,
                breaker: Mutex::new(LdapCircuitBreaker::new(3, Duration::from_secs(30))),
            })
            .collect();

        Self {
            entries,
            read_policy: ReadPolicy::RoundRobin,
            rr_cursor: AtomicUsize::new(0),
            transport: Transport::Real,
        }
    }

    /// Create a pool backed by a mock authentication function.
    ///
    /// The closure receives `(server_uri, username, password)` and returns
    /// `true` iff the credentials are valid for that server.
    pub fn with_mock_transport<F>(mut servers: Vec<LdapServer>, auth_fn: F) -> Self
    where
        F: Fn(&str, &str, &str) -> bool + Send + Sync + 'static,
    {
        servers.sort_by_key(|s| s.priority);
        let entries = servers
            .into_iter()
            .map(|s| PoolEntry {
                server: s,
                breaker: Mutex::new(LdapCircuitBreaker::new(3, Duration::from_secs(30))),
            })
            .collect();

        Self {
            entries,
            read_policy: ReadPolicy::RoundRobin,
            rr_cursor: AtomicUsize::new(0),
            transport: Transport::Mock(Arc::new(auth_fn)),
        }
    }

    /// Create a pool with a custom read policy.
    pub fn with_read_policy(mut self, policy: ReadPolicy) -> Self {
        self.read_policy = policy;
        self
    }

    /// Override the circuit-breaker configuration for all servers.
    pub fn with_circuit_breaker(self, threshold: u32, reset_timeout: Duration) -> Self {
        for entry in &self.entries {
            let mut breaker = entry.breaker.lock().expect("breaker mutex poisoned");
            *breaker = LdapCircuitBreaker::new(threshold, reset_timeout);
        }
        self
    }

    // ── Server selection ─────────────────────────────────────────────────────

    /// Select a server URI for a **read** operation according to the pool
    /// policy.  Returns `None` if all servers have open circuit breakers.
    pub fn select_read_server(&self) -> Option<&str> {
        match self.read_policy {
            ReadPolicy::RoundRobin => self.round_robin_read(),
            ReadPolicy::PrimaryPreferred => self.primary_preferred_read(),
        }
    }

    /// Select the **primary** server URI for write / bind operations.
    /// Returns `None` if the primary circuit breaker is open.
    pub fn select_write_server(&self) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| {
                e.server.role == LdapServerRole::Primary
                    && !e.breaker.lock().expect("breaker mutex poisoned").is_open()
            })
            .map(|e| e.server.uri.as_str())
    }

    // ── Health / circuit-breaker management ──────────────────────────────────

    /// Mark a server as having failed.  Increments its circuit-breaker counter.
    pub fn record_failure(&self, uri: &str) {
        if let Some(entry) = self.entries.iter().find(|e| e.server.uri == uri) {
            entry
                .breaker
                .lock()
                .expect("breaker mutex poisoned")
                .record_failure();
        }
    }

    /// Mark a server as healthy.  Resets its circuit-breaker counter.
    pub fn record_success(&self, uri: &str) {
        if let Some(entry) = self.entries.iter().find(|e| e.server.uri == uri) {
            entry
                .breaker
                .lock()
                .expect("breaker mutex poisoned")
                .record_success();
        }
    }

    /// Return a snapshot of each server's health status.
    pub fn health_status(&self) -> Vec<(String, LdapServerHealth)> {
        self.entries
            .iter()
            .map(|e| {
                let health = e.breaker.lock().expect("breaker mutex poisoned").health();
                (e.server.uri.clone(), health)
            })
            .collect()
    }

    // ── High-level operations with automatic failover ─────────────────────────

    /// Attempt a bind (authenticate) against the **primary** server first,
    /// then replicas in priority order, skipping servers whose circuit breaker
    /// is open.
    ///
    /// Failures are recorded on each attempted server; success resets the
    /// counter for the winning server.
    pub async fn bind_with_failover(
        &self,
        username: &str,
        password: &str,
    ) -> LdapResult<LdapBoundSession> {
        let mut tried: Vec<String> = Vec::new();
        let mut last_reason = String::new();

        for entry in &self.entries {
            let uri = &entry.server.uri;

            // Skip servers with open circuit breakers.
            if entry
                .breaker
                .lock()
                .expect("breaker mutex poisoned")
                .is_open()
            {
                continue;
            }

            tried.push(uri.clone());

            let success = self.do_bind(uri, username, password).await;
            if success {
                entry
                    .breaker
                    .lock()
                    .expect("breaker mutex poisoned")
                    .record_success();
                return Ok(LdapBoundSession {
                    server_uri: uri.clone(),
                    username: username.to_string(),
                });
            } else {
                entry
                    .breaker
                    .lock()
                    .expect("breaker mutex poisoned")
                    .record_failure();
                last_reason = format!("bind rejected by {uri}");
            }
        }

        if tried.is_empty() {
            return Err(LdapHaError::NoServersAvailable);
        }

        // If we tried at least one server but all said "no", distinguish
        // between a transport-level failure (AllServersUnreachable) and a
        // credential failure (AuthFailed).  Since our stub always rejects
        // on bad credentials rather than on a network error, return AuthFailed.
        Err(LdapHaError::AuthFailed {
            reason: if last_reason.is_empty() {
                "all servers rejected bind".to_string()
            } else {
                last_reason
            },
        })
    }

    /// Perform a search with automatic failover.
    ///
    /// Tries healthy replicas first (in priority order), then the primary,
    /// recording failures along the way.
    pub async fn search_with_failover(
        &self,
        base_dn: &str,
        filter: &str,
    ) -> LdapResult<Vec<LdapEntry>> {
        let mut tried: Vec<String> = Vec::new();

        // Build a candidate list: replicas first, then primaries — all
        // filtered to healthy servers only.
        let candidates: Vec<&PoolEntry> = {
            let mut replicas: Vec<&PoolEntry> = self
                .entries
                .iter()
                .filter(|e| {
                    e.server.role == LdapServerRole::Replica
                        && !e.breaker.lock().expect("breaker mutex poisoned").is_open()
                })
                .collect();
            let mut primaries: Vec<&PoolEntry> = self
                .entries
                .iter()
                .filter(|e| {
                    e.server.role == LdapServerRole::Primary
                        && !e.breaker.lock().expect("breaker mutex poisoned").is_open()
                })
                .collect();
            replicas.append(&mut primaries);
            replicas
        };

        if candidates.is_empty() {
            return Err(LdapHaError::NoServersAvailable);
        }

        for entry in candidates {
            let uri = &entry.server.uri;
            tried.push(uri.clone());

            match self.do_search(uri, base_dn, filter).await {
                Some(results) => {
                    entry
                        .breaker
                        .lock()
                        .expect("breaker mutex poisoned")
                        .record_success();
                    return Ok(results);
                }
                None => {
                    entry
                        .breaker
                        .lock()
                        .expect("breaker mutex poisoned")
                        .record_failure();
                }
            }
        }

        Err(LdapHaError::AllServersUnreachable { servers: tried })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Perform a bind operation against the given URI.
    /// Returns `true` on success.  In production this would call a real LDAP
    /// client; in test mode the mock closure is used.
    async fn do_bind(&self, uri: &str, username: &str, password: &str) -> bool {
        match &self.transport {
            Transport::Real => {
                // Stub: a real implementation would use an async LDAP client.
                // Succeed unless password is empty (mirrors the existing ldap.rs stub).
                !password.is_empty() && !username.is_empty()
            }
            Transport::Mock(auth_fn) => auth_fn(uri, username, password),
        }
    }

    /// Perform a search operation against the given URI.
    /// Returns `None` on transport failure (triggers failover).
    async fn do_search(&self, uri: &str, base_dn: &str, filter: &str) -> Option<Vec<LdapEntry>> {
        match &self.transport {
            Transport::Real => {
                // Stub: return a synthetic entry so callers can unit-test
                // without a real LDAP directory.
                let mut attrs = HashMap::new();
                attrs.insert("cn".to_string(), vec!["stub".to_string()]);
                Some(vec![LdapEntry {
                    dn: format!("cn=stub,{base_dn}"),
                    attrs,
                }])
            }
            Transport::Mock(_) => {
                // For mock transport, simulate a successful (empty) search on
                // any server that has not been poisoned externally.  The
                // caller tests failover by driving circuit-breaker state
                // directly via `record_failure`.
                let _ = (uri, filter); // suppress unused-variable warning
                let mut attrs = HashMap::new();
                attrs.insert("cn".to_string(), vec!["mockuser".to_string()]);
                Some(vec![LdapEntry {
                    dn: format!("cn=mockuser,{base_dn}"),
                    attrs,
                }])
            }
        }
    }

    /// Round-robin selection across all healthy servers (any role).
    fn round_robin_read(&self) -> Option<&str> {
        let healthy: Vec<&PoolEntry> = self
            .entries
            .iter()
            .filter(|e| !e.breaker.lock().expect("breaker mutex poisoned").is_open())
            .collect();

        if healthy.is_empty() {
            return None;
        }

        // Atomically advance the cursor and wrap around the healthy count.
        let idx = self.rr_cursor.fetch_add(1, Ordering::Relaxed) % healthy.len();
        Some(healthy[idx].server.uri.as_str())
    }

    /// Primary-preferred selection: return the primary if healthy, else the
    /// first healthy replica.
    fn primary_preferred_read(&self) -> Option<&str> {
        // Try primary first.
        if let Some(entry) = self.entries.iter().find(|e| {
            e.server.role == LdapServerRole::Primary
                && !e.breaker.lock().expect("breaker mutex poisoned").is_open()
        }) {
            return Some(entry.server.uri.as_str());
        }

        // Fall back to first healthy replica.
        self.entries
            .iter()
            .find(|e| {
                e.server.role == LdapServerRole::Replica
                    && !e.breaker.lock().expect("breaker mutex poisoned").is_open()
            })
            .map(|e| e.server.uri.as_str())
    }
}

// ─── Helper constructor for common configurations ─────────────────────────────

/// Build a pool from a primary URI and zero or more replica URIs.
///
/// The primary receives priority 0; replicas receive priorities 1, 2, …
pub fn build_ha_pool(primary_uri: &str, replica_uris: &[&str]) -> LdapHaPool {
    let mut servers = vec![LdapServer {
        uri: primary_uri.to_string(),
        role: LdapServerRole::Primary,
        priority: 0,
    }];
    for (i, uri) in replica_uris.iter().enumerate() {
        servers.push(LdapServer {
            uri: uri.to_string(),
            role: LdapServerRole::Replica,
            priority: (i + 1) as u8,
        });
    }
    LdapHaPool::new(servers)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Convenience: build a mock pool where every server accepts any non-empty
    /// password.
    fn mock_pool_accept_nonempty(primary: &str, replicas: &[&str]) -> LdapHaPool {
        let mut servers = vec![LdapServer {
            uri: primary.to_string(),
            role: LdapServerRole::Primary,
            priority: 0,
        }];
        for (i, r) in replicas.iter().enumerate() {
            servers.push(LdapServer {
                uri: r.to_string(),
                role: LdapServerRole::Replica,
                priority: (i + 1) as u8,
            });
        }
        LdapHaPool::with_mock_transport(servers, |_uri, _user, pass| !pass.is_empty())
    }

    // ── Test 1: single primary — select_write_server ──────────────────────────

    #[test]
    fn test_single_primary_write_server() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &[]);
        let uri = pool.select_write_server();
        assert_eq!(uri, Some("ldap://primary:389"));
    }

    // ── Test 2: primary + replica — select_read_server round-robins ──────────

    #[test]
    fn test_round_robin_includes_both_servers() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &["ldap://replica:389"]);

        // Collect a window of round-robin selections.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for _ in 0..8 {
            if let Some(uri) = pool.select_read_server() {
                seen.insert(uri.to_string());
            }
        }

        assert!(seen.contains("ldap://primary:389"));
        assert!(seen.contains("ldap://replica:389"));
    }

    // ── Test 3: failed server is skipped in select_read_server ───────────────

    #[test]
    fn test_failed_server_skipped_in_read() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &["ldap://replica:389"])
            .with_circuit_breaker(1, Duration::from_secs(3600));

        // Exhaust the primary's circuit breaker.
        pool.record_failure("ldap://primary:389"); // count=1, threshold=1 → open

        // All reads should now land on the replica only.
        for _ in 0..10 {
            let uri = pool
                .select_read_server()
                .expect("replica should still be available");
            assert_eq!(
                uri, "ldap://replica:389",
                "open-circuit primary must be skipped"
            );
        }
    }

    // ── Test 4: all servers unreachable → NoServersAvailable ─────────────────

    #[test]
    fn test_all_unreachable_returns_none() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &["ldap://replica:389"])
            .with_circuit_breaker(1, Duration::from_secs(3600));

        pool.record_failure("ldap://primary:389");
        pool.record_failure("ldap://replica:389");

        assert!(pool.select_read_server().is_none());
        assert!(pool.select_write_server().is_none());
    }

    // ── Test 5: circuit breaker opens after threshold failures ────────────────

    #[test]
    fn test_circuit_breaker_opens_at_threshold() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &[])
            .with_circuit_breaker(3, Duration::from_secs(3600));

        // Two failures: still closed.
        pool.record_failure("ldap://primary:389");
        pool.record_failure("ldap://primary:389");
        assert!(
            pool.select_write_server().is_some(),
            "below threshold — still healthy"
        );

        // Third failure: circuit opens.
        pool.record_failure("ldap://primary:389");
        assert!(
            pool.select_write_server().is_none(),
            "at threshold — circuit must be open"
        );
    }

    // ── Test 6: circuit breaker resets after reset_timeout ───────────────────

    #[test]
    fn test_circuit_breaker_resets_after_timeout() {
        // Use a very short reset timeout so the test doesn't actually sleep.
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &[])
            .with_circuit_breaker(1, Duration::from_millis(1));

        pool.record_failure("ldap://primary:389"); // opens

        // Spin until the reset window has definitely elapsed (should be < 10 ms).
        let deadline = std::time::Instant::now() + Duration::from_millis(200);
        while std::time::Instant::now() < deadline {
            std::hint::spin_loop();
        }

        // After reset_timeout the breaker should auto-clear.
        assert!(
            pool.select_write_server().is_some(),
            "circuit breaker should reset after timeout"
        );
    }

    // ── Test 7: bind_with_failover succeeds on second server ─────────────────

    #[tokio::test]
    async fn test_bind_failover_to_second_server() {
        // First server always rejects; second always accepts.
        let servers = vec![
            LdapServer {
                uri: "ldap://server1:389".to_string(),
                role: LdapServerRole::Primary,
                priority: 0,
            },
            LdapServer {
                uri: "ldap://server2:389".to_string(),
                role: LdapServerRole::Replica,
                priority: 1,
            },
        ];

        let pool = LdapHaPool::with_mock_transport(servers, |uri, _user, pass| {
            // server1 always fails; server2 accepts non-empty passwords.
            !uri.contains("server1") && !pass.is_empty()
        });

        let session = pool
            .bind_with_failover("alice", "secret")
            .await
            .expect("should succeed on server2");

        assert_eq!(session.server_uri, "ldap://server2:389");
        assert_eq!(session.username, "alice");
    }

    // ── Test 8: health_status shows Unreachable for open-circuit server ───────

    #[test]
    fn test_health_status_unreachable() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &["ldap://replica:389"])
            .with_circuit_breaker(2, Duration::from_secs(3600));

        pool.record_failure("ldap://primary:389");
        pool.record_failure("ldap://primary:389"); // opens

        let statuses: HashMap<String, LdapServerHealth> =
            pool.health_status().into_iter().collect();

        assert_eq!(
            statuses.get("ldap://primary:389").copied(),
            Some(LdapServerHealth::Unreachable)
        );
        assert_eq!(
            statuses.get("ldap://replica:389").copied(),
            Some(LdapServerHealth::Healthy)
        );
    }

    // ── Test 9: PrimaryPreferred reads from primary when healthy ─────────────

    #[test]
    fn test_primary_preferred_reads_from_primary() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &["ldap://replica:389"])
            .with_read_policy(ReadPolicy::PrimaryPreferred);

        // With primary healthy, every read should go to the primary.
        for _ in 0..10 {
            let uri = pool.select_read_server().expect("primary is healthy");
            assert_eq!(uri, "ldap://primary:389");
        }
    }

    // ── Test 10: round-robin cycles through healthy replicas in order ─────────

    #[test]
    fn test_round_robin_cycles_through_replicas() {
        // Two replicas; trip the primary so only replicas are available.
        let pool = mock_pool_accept_nonempty(
            "ldap://primary:389",
            &["ldap://replica1:389", "ldap://replica2:389"],
        )
        .with_circuit_breaker(1, Duration::from_secs(3600));

        pool.record_failure("ldap://primary:389"); // open

        // Collect 6 selections — should alternate between the two replicas.
        let selections: Vec<String> = (0..6)
            .filter_map(|_| pool.select_read_server().map(str::to_string))
            .collect();

        assert!(!selections.is_empty());
        // Both replicas must appear.
        assert!(selections.iter().any(|s| s == "ldap://replica1:389"));
        assert!(selections.iter().any(|s| s == "ldap://replica2:389"));
        // Primary must NOT appear.
        assert!(!selections.iter().any(|s| s == "ldap://primary:389"));
    }

    // ── Bonus test 11: bind_with_failover fails when all servers reject ───────

    #[tokio::test]
    async fn test_bind_fails_when_all_reject() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &["ldap://replica:389"]);

        // Empty password → all servers reject.
        let err = pool
            .bind_with_failover("alice", "")
            .await
            .expect_err("should fail");

        assert!(
            matches!(err, LdapHaError::AuthFailed { .. }),
            "expected AuthFailed, got: {err}"
        );
    }

    // ── Bonus test 12: search_with_failover returns NoServersAvailable ────────

    #[tokio::test]
    async fn test_search_fails_when_all_unreachable() {
        let pool = mock_pool_accept_nonempty("ldap://primary:389", &["ldap://replica:389"])
            .with_circuit_breaker(1, Duration::from_secs(3600));

        pool.record_failure("ldap://primary:389");
        pool.record_failure("ldap://replica:389");

        let err = pool
            .search_with_failover("dc=example,dc=com", "(uid=alice)")
            .await
            .expect_err("should fail");

        assert!(
            matches!(err, LdapHaError::NoServersAvailable),
            "expected NoServersAvailable, got: {err}"
        );
    }
}
