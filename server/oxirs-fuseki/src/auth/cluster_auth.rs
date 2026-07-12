//! Cross-node authentication tokens for OxiRS Fuseki cluster communication.
//!
//! When Fuseki nodes in a cluster communicate with each other, they must
//! authenticate so that only legitimate cluster members can call internal
//! admin/federation APIs. This module implements short-lived, HMAC-SHA256-signed
//! bearer tokens for that purpose.
//!
//! ## Token format
//!
//! ```text
//! base64url(JSON payload) "." hex(HMAC-SHA256 signature)
//! ```
//!
//! The payload is a JSON object containing `node_id`, `issued_at`, `expires_at`,
//! and `jti`. The signature covers the canonical string
//! `"{node_id}|{issued_at}|{expires_at}|{jti}"`.
//!
//! ## Security properties
//!
//! | Property          | Mechanism                                          |
//! |-------------------|----------------------------------------------------|
//! | Unforgeability    | HMAC-SHA256 over canonical message, 32-byte key    |
//! | Short-lived       | Default 300 s TTL, checked on every verification  |
//! | Replay-resistance | Unique per-token JTI; explicit revocation list     |
//! | Clock-skew        | Configurable slack window (default 30 s)           |
//! | Secret rotation   | `rotate_secret` swaps key; all old tokens invalid  |

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during cluster authentication operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ClusterAuthError {
    /// The node ID is not registered in this manager.
    #[error("Unknown node: {0}")]
    UnknownNode(String),

    /// The token's `expires_at` timestamp has passed.
    #[error("Token expired at unix timestamp {expired_at}")]
    TokenExpired {
        /// Unix timestamp (seconds) when the token expired.
        expired_at: u64,
    },

    /// The HMAC signature does not match the token payload.
    #[error("Invalid token signature")]
    InvalidSignature,

    /// The JTI has been explicitly revoked.
    #[error("Token has been revoked: {0}")]
    RevokedToken(String),

    /// Encoding or decoding of the bearer string failed.
    #[error("Token serialization error: {0}")]
    SerializationError(String),
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Identifies a specific cluster node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeIdentity {
    /// UUID-style identifier, e.g. `"node-a1b2c3"`.
    pub node_id: String,
    /// Human-readable display name.
    pub display_name: String,
    /// Unix timestamp (seconds) when the node was registered.
    pub registered_at: u64,
}

/// A signed, short-lived token issued to a cluster node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClusterNodeToken {
    /// The node ID this token was issued to.
    pub node_id: String,
    /// Unix timestamp (seconds) when the token was issued.
    pub issued_at: u64,
    /// Unix timestamp (seconds) when the token expires.
    pub expires_at: u64,
    /// Unique token ID (random 16-byte value, hex-encoded).
    pub jti: String,
    /// Hex-encoded HMAC-SHA256 over the canonical signing message.
    pub signature: String,
}

/// Configuration for the cluster authentication system.
#[derive(Debug, Clone)]
pub struct ClusterAuthConfig {
    /// How long issued tokens remain valid (seconds). Default: 300.
    pub token_ttl_seconds: u64,
    /// Maximum allowed clock skew between nodes (seconds). Default: 30.
    pub max_clock_skew_seconds: u64,
}

impl Default for ClusterAuthConfig {
    fn default() -> Self {
        Self {
            token_ttl_seconds: 300,
            max_clock_skew_seconds: 30,
        }
    }
}

// ---------------------------------------------------------------------------
// Manager
// ---------------------------------------------------------------------------

/// Manages cluster node registration, token issuance, and verification.
///
/// The manager is **not** `Clone` because it holds the mutable revocation list
/// and node registry. Wrap in `Arc<RwLock<…>>` for shared, concurrent access
/// across Axum handlers.
pub struct ClusterAuthManager {
    config: ClusterAuthConfig,
    nodes: HashMap<String, NodeIdentity>,
    /// HMAC-SHA256 signing key bytes. Stored as raw bytes and passed to the
    /// stateless `oxicrypto-mac` one-shot HMAC on demand.
    secret: Vec<u8>,
    revoked_jtis: HashSet<String>,
}

impl ClusterAuthManager {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Create a new manager with the given configuration and signing secret.
    ///
    /// `secret` should be a high-entropy byte slice of at least 32 bytes.
    pub fn new(config: ClusterAuthConfig, secret: &[u8]) -> Self {
        Self {
            config,
            nodes: HashMap::new(),
            secret: secret.to_vec(),
            revoked_jtis: HashSet::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Node registry
    // -----------------------------------------------------------------------

    /// Register a new node identity.
    ///
    /// Returns `Err(UnknownNode)` … actually always succeeds; the error variant
    /// is reserved for potential future duplicate-detection policy.
    pub fn register_node(&mut self, node: NodeIdentity) -> Result<(), ClusterAuthError> {
        self.nodes.insert(node.node_id.clone(), node);
        Ok(())
    }

    /// Remove a node from the registry, returning its identity if it existed.
    pub fn remove_node(&mut self, node_id: &str) -> Option<NodeIdentity> {
        self.nodes.remove(node_id)
    }

    /// Return references to all registered nodes.
    pub fn known_nodes(&self) -> Vec<&NodeIdentity> {
        self.nodes.values().collect()
    }

    // -----------------------------------------------------------------------
    // Token issuance
    // -----------------------------------------------------------------------

    /// Issue a signed token for the given registered node.
    ///
    /// `now` is the current Unix timestamp in seconds. Tokens are valid from
    /// `now` to `now + config.token_ttl_seconds`.
    ///
    /// # Errors
    /// - [`ClusterAuthError::UnknownNode`] if `node_id` is not registered.
    pub fn issue_token(
        &self,
        node_id: &str,
        now: u64,
    ) -> Result<ClusterNodeToken, ClusterAuthError> {
        if !self.nodes.contains_key(node_id) {
            return Err(ClusterAuthError::UnknownNode(node_id.to_string()));
        }

        let issued_at = now;
        let expires_at = now + self.config.token_ttl_seconds;
        let jti = generate_jti();

        let signature = self.compute_signature(node_id, issued_at, expires_at, &jti)?;

        Ok(ClusterNodeToken {
            node_id: node_id.to_string(),
            issued_at,
            expires_at,
            jti,
            signature,
        })
    }

    // -----------------------------------------------------------------------
    // Token verification
    // -----------------------------------------------------------------------

    /// Verify a cluster token and return a reference to the issuing node.
    ///
    /// Verification checks (in order):
    /// 1. HMAC signature integrity.
    /// 2. JTI revocation list.
    /// 3. Expiry (`expires_at + max_clock_skew` must be ≥ `now`).
    /// 4. Node is still registered.
    ///
    /// # Errors
    /// - [`ClusterAuthError::InvalidSignature`] – tampered payload.
    /// - [`ClusterAuthError::RevokedToken`] – JTI is revoked.
    /// - [`ClusterAuthError::TokenExpired`] – token lifetime elapsed.
    /// - [`ClusterAuthError::UnknownNode`] – node no longer registered.
    pub fn verify_token(
        &self,
        token: &ClusterNodeToken,
        now: u64,
    ) -> Result<&NodeIdentity, ClusterAuthError> {
        // 1. Verify signature first to prevent timing-based probing of the
        //    revocation list or node registry.
        let expected_sig = self.compute_signature(
            &token.node_id,
            token.issued_at,
            token.expires_at,
            &token.jti,
        )?;
        if !constant_time_eq(&expected_sig, &token.signature) {
            return Err(ClusterAuthError::InvalidSignature);
        }

        // 2. Check revocation list.
        if self.revoked_jtis.contains(&token.jti) {
            return Err(ClusterAuthError::RevokedToken(token.jti.clone()));
        }

        // 3. Check expiry with clock-skew allowance.
        let effective_expiry = token
            .expires_at
            .saturating_add(self.config.max_clock_skew_seconds);
        if now > effective_expiry {
            return Err(ClusterAuthError::TokenExpired {
                expired_at: token.expires_at,
            });
        }

        // 4. Confirm the node is still in the registry.
        self.nodes
            .get(&token.node_id)
            .ok_or_else(|| ClusterAuthError::UnknownNode(token.node_id.clone()))
    }

    // -----------------------------------------------------------------------
    // Revocation and secret rotation
    // -----------------------------------------------------------------------

    /// Add the given JTI to the revocation list so it is always rejected.
    pub fn revoke_token(&mut self, jti: &str) {
        self.revoked_jtis.insert(jti.to_string());
    }

    /// Replace the HMAC signing secret.
    ///
    /// All tokens signed with the previous secret become invalid immediately
    /// because `verify_token` will recompute signatures using the new key.
    pub fn rotate_secret(&mut self, new_secret: &[u8]) {
        self.secret = new_secret.to_vec();
    }

    // -----------------------------------------------------------------------
    // Bearer encoding / decoding
    // -----------------------------------------------------------------------

    /// Encode a token into a bearer string suitable for HTTP `Authorization`
    /// headers:
    ///
    /// ```text
    /// base64url(JSON payload) "." hex(signature)
    /// ```
    pub fn encode_token(token: &ClusterNodeToken) -> String {
        // Serialize the payload (everything except the signature).
        let payload = ClusterNodePayload {
            node_id: token.node_id.clone(),
            issued_at: token.issued_at,
            expires_at: token.expires_at,
            jti: token.jti.clone(),
        };
        let json = serde_json::to_string(&payload).unwrap_or_default();
        let encoded_payload = URL_SAFE_NO_PAD.encode(json.as_bytes());
        format!("{}.{}", encoded_payload, token.signature)
    }

    /// Decode a bearer string produced by [`Self::encode_token`] back into a
    /// [`ClusterNodeToken`].
    ///
    /// # Errors
    /// - [`ClusterAuthError::SerializationError`] if the bearer value is
    ///   malformed (wrong format, invalid base64, invalid JSON).
    pub fn decode_token(bearer: &str) -> Result<ClusterNodeToken, ClusterAuthError> {
        let (encoded_payload, signature) = bearer
            .rsplit_once('.')
            .ok_or_else(|| ClusterAuthError::SerializationError("Missing '.' separator".into()))?;

        let json_bytes = URL_SAFE_NO_PAD
            .decode(encoded_payload)
            .map_err(|e| ClusterAuthError::SerializationError(format!("Base64 decode: {e}")))?;

        let json_str = std::str::from_utf8(&json_bytes)
            .map_err(|e| ClusterAuthError::SerializationError(format!("UTF-8 decode: {e}")))?;

        let payload: ClusterNodePayload = serde_json::from_str(json_str)
            .map_err(|e| ClusterAuthError::SerializationError(format!("JSON parse: {e}")))?;

        Ok(ClusterNodeToken {
            node_id: payload.node_id,
            issued_at: payload.issued_at,
            expires_at: payload.expires_at,
            jti: payload.jti,
            signature: signature.to_string(),
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Compute HMAC-SHA256 over the canonical signing message and return the
    /// result as a lowercase hex string.
    ///
    /// Canonical message: `"{node_id}|{issued_at}|{expires_at}|{jti}"`
    ///
    /// Uses the Pure-Rust `oxicrypto-mac` one-shot HMAC-SHA256. The computation
    /// is identical to the previous `ring::hmac` implementation (same key, same
    /// message bytes), so existing cluster tokens remain valid.
    fn compute_signature(
        &self,
        node_id: &str,
        issued_at: u64,
        expires_at: u64,
        jti: &str,
    ) -> Result<String, ClusterAuthError> {
        let message = format!("{node_id}|{issued_at}|{expires_at}|{jti}");
        let tag = oxicrypto_mac::hmac_sha256_to_vec(&self.secret, message.as_bytes())
            .map_err(|e| ClusterAuthError::SerializationError(format!("HMAC error: {e:?}")))?;
        Ok(hex::encode(tag))
    }
}

// ---------------------------------------------------------------------------
// Intermediate serde type (payload without signature)
// ---------------------------------------------------------------------------

/// JSON payload embedded in the bearer token (excludes the signature field).
#[derive(Debug, Serialize, Deserialize)]
struct ClusterNodePayload {
    node_id: String,
    issued_at: u64,
    expires_at: u64,
    jti: String,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Generate a unique token ID as 16 random bytes, hex-encoded (32 hex chars).
///
/// Uses `scirs2_core::random::SecureRandom` to avoid direct `rand` dependency.
fn generate_jti() -> String {
    use scirs2_core::random::SecureRandom;
    let mut secure = SecureRandom::new();
    let bytes = secure.random_bytes(16);
    hex::encode(&bytes)
}

/// Constant-time string comparison to avoid timing attacks when checking
/// HMAC outputs.
///
/// Both strings must be hex-encoded HMAC tags, so they are ASCII and have the
/// same length when valid. We compare byte-by-byte accumulating differences
/// via XOR.
fn constant_time_eq(a: &str, b: &str) -> bool {
    let a = a.as_bytes();
    let b = b.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let diff = a
        .iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y));
    diff == 0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Shared test helpers -------------------------------------------------------

    fn test_secret() -> Vec<u8> {
        b"test-cluster-secret-32-bytes-long!!".to_vec()
    }

    fn test_config() -> ClusterAuthConfig {
        ClusterAuthConfig {
            token_ttl_seconds: 300,
            max_clock_skew_seconds: 30,
        }
    }

    fn test_node(suffix: &str) -> NodeIdentity {
        NodeIdentity {
            node_id: format!("node-{suffix}"),
            display_name: format!("Test Node {suffix}"),
            registered_at: 1_700_000_000,
        }
    }

    fn manager_with_node(suffix: &str) -> (ClusterAuthManager, String) {
        let mut mgr = ClusterAuthManager::new(test_config(), &test_secret());
        let node = test_node(suffix);
        let node_id = node.node_id.clone();
        mgr.register_node(node).unwrap();
        (mgr, node_id)
    }

    // -----------------------------------------------------------------------
    // 1. Register and issue token
    // -----------------------------------------------------------------------
    #[test]
    fn test_register_and_issue_token() {
        let (mgr, node_id) = manager_with_node("a1b2c3");
        let now: u64 = 1_700_100_000;
        let token = mgr.issue_token(&node_id, now).expect("issue_token");

        assert_eq!(token.node_id, node_id);
        assert_eq!(token.issued_at, now);
        assert_eq!(token.expires_at, now + 300);
        assert!(!token.jti.is_empty());
        assert!(!token.signature.is_empty());

        // Token should verify cleanly.
        let identity = mgr.verify_token(&token, now).expect("verify_token");
        assert_eq!(identity.node_id, node_id);
    }

    // -----------------------------------------------------------------------
    // 2. Token expiry
    // -----------------------------------------------------------------------
    #[test]
    fn test_token_expiry() {
        let (mgr, node_id) = manager_with_node("exp");
        let now: u64 = 1_700_200_000;
        let token = mgr.issue_token(&node_id, now).expect("issue_token");

        // Advance time beyond TTL + clock skew.
        let future = now + 300 + 30 + 1; // expires_at + max_clock_skew + 1
        let err = mgr
            .verify_token(&token, future)
            .expect_err("should be expired");

        assert!(
            matches!(err, ClusterAuthError::TokenExpired { expired_at } if expired_at == now + 300),
            "expected TokenExpired, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Invalid (tampered) signature
    // -----------------------------------------------------------------------
    #[test]
    fn test_invalid_signature() {
        let (mgr, node_id) = manager_with_node("sig");
        let now: u64 = 1_700_300_000;
        let mut token = mgr.issue_token(&node_id, now).expect("issue_token");

        // Tamper with the signature (flip the last hex digit).
        let last = token.signature.pop().unwrap();
        let flipped = if last == '0' { '1' } else { '0' };
        token.signature.push(flipped);

        let err = mgr
            .verify_token(&token, now)
            .expect_err("tampered token should fail");
        assert_eq!(err, ClusterAuthError::InvalidSignature);
    }

    // -----------------------------------------------------------------------
    // 4. Issuing to unknown node returns UnknownNode
    // -----------------------------------------------------------------------
    #[test]
    fn test_unknown_node_issue() {
        let mgr = ClusterAuthManager::new(test_config(), &test_secret());
        let err = mgr
            .issue_token("node-ghost", 1_700_000_000)
            .expect_err("unknown node should fail");
        assert!(
            matches!(err, ClusterAuthError::UnknownNode(ref id) if id == "node-ghost"),
            "unexpected error: {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 5. Token for unknown node is rejected at verify time
    // -----------------------------------------------------------------------
    #[test]
    fn test_unknown_node_verify() {
        // Issue with full manager, then verify with an empty one.
        let (issuing_mgr, node_id) = manager_with_node("ghost2");
        let now: u64 = 1_700_000_000;
        let token = issuing_mgr.issue_token(&node_id, now).unwrap();

        // Verifying manager does NOT have the node registered.
        let verifying_mgr = ClusterAuthManager::new(test_config(), &test_secret());
        let err = verifying_mgr
            .verify_token(&token, now)
            .expect_err("unregistered node should fail");
        assert!(
            matches!(err, ClusterAuthError::UnknownNode(_)),
            "unexpected error: {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 6. Revoked JTI is rejected
    // -----------------------------------------------------------------------
    #[test]
    fn test_revoke_token() {
        let (mut mgr, node_id) = manager_with_node("rev");
        let now: u64 = 1_700_400_000;
        let token = mgr.issue_token(&node_id, now).unwrap();
        let jti = token.jti.clone();

        mgr.revoke_token(&jti);

        let err = mgr
            .verify_token(&token, now)
            .expect_err("revoked token should fail");
        assert!(
            matches!(err, ClusterAuthError::RevokedToken(ref j) if j == &jti),
            "unexpected error: {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 7. Token issued with old secret is rejected after rotation
    // -----------------------------------------------------------------------
    #[test]
    fn test_rotate_secret() {
        let (mut mgr, node_id) = manager_with_node("rot");
        let now: u64 = 1_700_500_000;
        let token = mgr.issue_token(&node_id, now).unwrap();

        // Token verifies with original secret.
        mgr.verify_token(&token, now)
            .expect("should verify before rotation");

        // Rotate to a different secret.
        let new_secret = b"new-completely-different-secret!!";
        mgr.rotate_secret(new_secret);

        // The old token's signature no longer matches.
        let err = mgr
            .verify_token(&token, now)
            .expect_err("old token should be invalid after rotation");
        assert_eq!(err, ClusterAuthError::InvalidSignature);
    }

    // -----------------------------------------------------------------------
    // 8. Encode / decode round-trip preserves all fields
    // -----------------------------------------------------------------------
    #[test]
    fn test_encode_decode_roundtrip() {
        let (mgr, node_id) = manager_with_node("enc");
        let now: u64 = 1_700_600_000;
        let original = mgr.issue_token(&node_id, now).unwrap();

        let bearer = ClusterAuthManager::encode_token(&original);
        let decoded = ClusterAuthManager::decode_token(&bearer).expect("decode_token");

        assert_eq!(decoded, original);
    }

    // -----------------------------------------------------------------------
    // 9. Clock-skew allowance: token is accepted when now is within skew
    // -----------------------------------------------------------------------
    #[test]
    fn test_clock_skew_allowed() {
        let (mgr, node_id) = manager_with_node("skew");
        let now: u64 = 1_700_700_000;
        let token = mgr.issue_token(&node_id, now).unwrap();

        // `now + 300` is exactly at expiry; add skew-1 so we are still inside.
        let skewed_now = now + 300 + 30 - 1; // expires_at + max_clock_skew - 1
        mgr.verify_token(&token, skewed_now)
            .expect("should be accepted within skew window");

        // One second beyond the skew window must fail.
        let over_skew = now + 300 + 30 + 1;
        let err = mgr
            .verify_token(&token, over_skew)
            .expect_err("should be rejected beyond skew window");
        assert!(matches!(err, ClusterAuthError::TokenExpired { .. }));
    }

    // -----------------------------------------------------------------------
    // 10. known_nodes() returns all registered nodes
    // -----------------------------------------------------------------------
    #[test]
    fn test_known_nodes_list() {
        let mut mgr = ClusterAuthManager::new(test_config(), &test_secret());
        mgr.register_node(test_node("n1")).unwrap();
        mgr.register_node(test_node("n2")).unwrap();
        mgr.register_node(test_node("n3")).unwrap();

        let nodes = mgr.known_nodes();
        assert_eq!(nodes.len(), 3);
    }

    // -----------------------------------------------------------------------
    // 11. Removed node is gone from registry; its token is rejected
    // -----------------------------------------------------------------------
    #[test]
    fn test_remove_node() {
        let (mut mgr, node_id) = manager_with_node("rm");
        let now: u64 = 1_700_800_000;
        let token = mgr.issue_token(&node_id, now).unwrap();

        // Remove the node.
        let removed = mgr.remove_node(&node_id);
        assert!(removed.is_some());
        assert!(!mgr.known_nodes().iter().any(|n| n.node_id == node_id));

        // The node's signature is correct but the node is no longer registered,
        // so verify_token must return UnknownNode.
        let err = mgr
            .verify_token(&token, now)
            .expect_err("removed node token should fail");
        assert!(
            matches!(err, ClusterAuthError::UnknownNode(_)),
            "unexpected error: {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // 12. ClusterAuthConfig::default() has expected TTL values
    // -----------------------------------------------------------------------
    #[test]
    fn test_config_defaults() {
        let cfg = ClusterAuthConfig::default();
        assert_eq!(cfg.token_ttl_seconds, 300);
        assert_eq!(cfg.max_clock_skew_seconds, 30);
    }
}
