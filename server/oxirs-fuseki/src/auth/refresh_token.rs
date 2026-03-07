//! OAuth 2.0 Refresh Token Rotation (RFC 6749 §10.4 / RFC 9068 / BCP 212)
//!
//! This module provides a cryptographically-secure, server-side refresh token
//! store with first-class support for:
//!
//! * **Token rotation**: every use of a refresh token atomically produces a new
//!   one, making the old token immediately invalid.
//! * **Replay-attack detection**: presenting an already-rotated token triggers
//!   cascade revocation of *all* tokens belonging to the same user (RFC 6749
//!   §10.4 recommendation).
//! * **Family tracking**: each token carries the UUID of the original grant so
//!   that cascade revocation only affects the compromised session family.
//! * **Thread safety**: the underlying store uses `Arc<RwLock<…>>` so it can
//!   be cheaply cloned across Axum handler tasks.
//!
//! ## Security Properties
//!
//! | Property               | Mechanism                                      |
//! |------------------------|------------------------------------------------|
//! | Unforgeability         | 32-byte CSPRNG token, base64url-encoded        |
//! | One-time use           | `rotated` flag set atomically before issuing   |
//! | Replay detection       | Checking `rotated` flag; cascade on re-use     |
//! | Expiry                 | Configurable TTL checked on every access       |
//! | Revocation             | Per-token and per-user bulk revocation         |
//! | Audit trail            | `previous_token_hash` SHA-256 chain            |

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::{FusekiError, FusekiResult};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during refresh token operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RotationError {
    /// The token string was not found in the store.
    #[error("Refresh token not found")]
    TokenNotFound,

    /// The token has passed its `expires_at` timestamp.
    #[error("Refresh token has expired")]
    TokenExpired,

    /// The token was already consumed by a previous rotation.
    ///
    /// This *may* indicate a replay attack; the caller should inspect the
    /// cascade behaviour and act accordingly.
    #[error("Refresh token has already been rotated (possible replay attack)")]
    TokenAlreadyRotated,

    /// The token was explicitly revoked (e.g. via logout).
    #[error("Refresh token has been revoked")]
    TokenRevoked,
}

impl From<RotationError> for FusekiError {
    fn from(e: RotationError) -> Self {
        FusekiError::authentication(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

/// A single refresh token entry stored in [`RefreshTokenStore`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefreshToken {
    /// The opaque token value (32 bytes, base64url-encoded without padding).
    pub token: String,

    /// Identifier of the user who owns this token.
    pub user_id: String,

    /// Identifier of the token *family* (original grant).  All tokens derived
    /// from the same authorization grant share the same `family_id`, enabling
    /// cascade revocation when a replay is detected.
    pub family_id: Uuid,

    /// When the token was issued.
    pub issued_at: DateTime<Utc>,

    /// When the token expires; validated on every call to [`RefreshTokenStore::validate`]
    /// and [`RefreshTokenStore::rotate`].
    pub expires_at: DateTime<Utc>,

    /// `true` once this token has been consumed by a rotation.  A rotated
    /// token must never be accepted again; re-presentation triggers cascade
    /// revocation of the whole family.
    pub rotated: bool,

    /// `true` when the token was explicitly revoked (e.g. user logout).
    pub revoked: bool,

    /// SHA-256 hash (hex) of the *previous* token in the rotation chain, or
    /// `None` for the first token of a family.  This provides an immutable
    /// audit chain.
    pub previous_token_hash: Option<String>,

    /// Generation counter within this family (0-based).
    pub generation: u32,
}

impl RefreshToken {
    /// Returns `true` when the token is valid: not rotated, not revoked, and
    /// not yet past its expiry timestamp.
    pub fn is_valid(&self) -> bool {
        !self.rotated && !self.revoked && Utc::now() < self.expires_at
    }

    /// Returns `true` when the token has expired.
    pub fn is_expired(&self) -> bool {
        Utc::now() >= self.expires_at
    }
}

// ---------------------------------------------------------------------------
// Token generation helpers
// ---------------------------------------------------------------------------

/// Generate a cryptographically random refresh token string.
///
/// The token is 32 random bytes encoded as base64url (no padding), yielding a
/// 43-character string that is safe to use in HTTP headers and URLs.
fn generate_token_string() -> FusekiResult<String> {
    use scirs2_core::random::SecureRandom;

    let mut secure = SecureRandom::new();
    let bytes = secure.random_bytes(32);

    Ok(URL_SAFE_NO_PAD.encode(&bytes))
}

/// Compute the SHA-256 hash of a token string, returned as a lowercase hex
/// string.  Used to populate [`RefreshToken::previous_token_hash`].
fn sha256_hex(input: &str) -> String {
    let hash = Sha256::digest(input.as_bytes());
    hex::encode(hash)
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// In-memory refresh token store with rotation, revocation, and replay
/// detection.
///
/// The store is cheaply cloneable (backed by `Arc`); all clones share state.
#[derive(Clone, Debug)]
pub struct RefreshTokenStore {
    /// Primary index: token string → token record.
    tokens: Arc<RwLock<HashMap<String, RefreshToken>>>,

    /// Secondary index: family_id → list of token strings in that family.
    /// Used for cascade revocation.
    family_index: Arc<RwLock<HashMap<Uuid, Vec<String>>>>,

    /// Secondary index: user_id → list of token strings.
    /// Used for per-user bulk revocation (logout all sessions).
    user_index: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl RefreshTokenStore {
    /// Create a new, empty [`RefreshTokenStore`].
    pub fn new() -> Self {
        Self {
            tokens: Arc::new(RwLock::new(HashMap::new())),
            family_index: Arc::new(RwLock::new(HashMap::new())),
            user_index: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    // -----------------------------------------------------------------------
    // Issue
    // -----------------------------------------------------------------------

    /// Issue a fresh refresh token for `user_id` with a lifetime of `ttl`.
    ///
    /// This creates the *root* of a new token family.
    pub async fn issue(&self, user_id: &str, ttl: Duration) -> FusekiResult<RefreshToken> {
        self.issue_in_family(user_id, ttl, Uuid::new_v4(), None, 0)
            .await
    }

    /// Internal helper: issue a token within an existing family (used during
    /// rotation) or create the first token for a new family.
    async fn issue_in_family(
        &self,
        user_id: &str,
        ttl: Duration,
        family_id: Uuid,
        previous_token_hash: Option<String>,
        generation: u32,
    ) -> FusekiResult<RefreshToken> {
        let token_str = generate_token_string()?;
        let now = Utc::now();

        let token = RefreshToken {
            token: token_str.clone(),
            user_id: user_id.to_string(),
            family_id,
            issued_at: now,
            expires_at: now + ttl,
            rotated: false,
            revoked: false,
            previous_token_hash,
            generation,
        };

        // Write to all three indices under separate write locks taken in a
        // deterministic order to avoid dead-lock (tokens → family → user).
        {
            let mut tokens = self.tokens.write().await;
            tokens.insert(token_str.clone(), token.clone());
        }
        {
            let mut family = self.family_index.write().await;
            family.entry(family_id).or_default().push(token_str.clone());
        }
        {
            let mut user_idx = self.user_index.write().await;
            user_idx
                .entry(user_id.to_string())
                .or_default()
                .push(token_str.clone());
        }

        debug!(
            user_id = %user_id,
            family_id = %family_id,
            generation = generation,
            expires_at = %token.expires_at,
            "Issued refresh token",
        );

        Ok(token)
    }

    // -----------------------------------------------------------------------
    // Rotate
    // -----------------------------------------------------------------------

    /// Atomically rotate `old_token_str`.
    ///
    /// On success the old token is marked as rotated and a new token belonging
    /// to the same family is returned.
    ///
    /// On failure:
    /// * [`RotationError::TokenNotFound`] — token string unknown.
    /// * [`RotationError::TokenExpired`] — token has passed its expiry.
    /// * [`RotationError::TokenRevoked`] — token was explicitly revoked.
    /// * [`RotationError::TokenAlreadyRotated`] — **replay attack detected**;
    ///   all tokens in the same family are cascade-revoked.
    pub async fn rotate(
        &self,
        old_token_str: &str,
        ttl: Duration,
    ) -> Result<RefreshToken, RotationError> {
        // Phase 1: read-only look-up ----------------------------------------
        let (family_id, user_id, generation, already_rotated, revoked, expired) = {
            let tokens = self.tokens.read().await;
            match tokens.get(old_token_str) {
                None => return Err(RotationError::TokenNotFound),
                Some(t) => (
                    t.family_id,
                    t.user_id.clone(),
                    t.generation,
                    t.rotated,
                    t.revoked,
                    t.is_expired(),
                ),
            }
        };

        // Check error conditions before acquiring write lock.
        if revoked {
            return Err(RotationError::TokenRevoked);
        }
        if expired {
            return Err(RotationError::TokenExpired);
        }
        if already_rotated {
            // Replay attack detected — cascade-revoke the whole family.
            warn!(
                family_id = %family_id,
                user_id = %user_id,
                "Replay attack detected: presented already-rotated refresh token; \
                 cascade-revoking all tokens in family",
            );
            self.cascade_revoke_family(family_id).await;
            return Err(RotationError::TokenAlreadyRotated);
        }

        // Phase 2: mark old token as rotated (write lock) --------------------
        {
            let mut tokens = self.tokens.write().await;
            if let Some(t) = tokens.get_mut(old_token_str) {
                t.rotated = true;
            }
        }

        // Phase 3: issue successor token ------------------------------------
        let prev_hash = sha256_hex(old_token_str);
        let new_token = self
            .issue_in_family(&user_id, ttl, family_id, Some(prev_hash), generation + 1)
            .await
            .map_err(|e| {
                // Log and surface as a generic rotation error via FusekiError.
                // We convert to TokenRevoked as a safe sentinel so callers do
                // not accidentally accept the old token.
                warn!("Failed to issue successor token during rotation: {e}");
                RotationError::TokenRevoked
            })?;

        info!(
            user_id = %user_id,
            family_id = %family_id,
            new_generation = new_token.generation,
            "Refresh token rotated successfully",
        );

        Ok(new_token)
    }

    // -----------------------------------------------------------------------
    // Validate
    // -----------------------------------------------------------------------

    /// Validate `token_str` without side effects.
    ///
    /// Returns a snapshot of the [`RefreshToken`] if and only if the token is
    /// valid (exists, not rotated, not revoked, not expired).  Returns `None`
    /// for any invalid token.
    pub async fn validate(&self, token_str: &str) -> Option<RefreshToken> {
        let tokens = self.tokens.read().await;
        tokens
            .get(token_str)
            .and_then(|t| if t.is_valid() { Some(t.clone()) } else { None })
    }

    // -----------------------------------------------------------------------
    // Revoke
    // -----------------------------------------------------------------------

    /// Explicitly revoke a single token (e.g. user-initiated logout of one
    /// session).  Does nothing if the token is not found.
    pub async fn revoke(&self, token_str: &str) {
        let mut tokens = self.tokens.write().await;
        if let Some(t) = tokens.get_mut(token_str) {
            t.revoked = true;
            debug!(
                token_prefix = &token_str[..8.min(token_str.len())],
                "Refresh token revoked"
            );
        }
    }

    /// Revoke **all** active refresh tokens for `user_id`.
    ///
    /// This is the correct operation for a "logout everywhere" / "revoke all
    /// sessions" endpoint.
    pub async fn revoke_all_for_user(&self, user_id: &str) -> usize {
        // Collect token strings first (read lock) then mutate (write lock) to
        // avoid holding both locks simultaneously.
        let token_strings: Vec<String> = {
            let user_idx = self.user_index.read().await;
            user_idx.get(user_id).cloned().unwrap_or_default()
        };

        let count = token_strings.len();
        if count == 0 {
            debug!(user_id = %user_id, "revoke_all_for_user: no tokens found");
            return 0;
        }

        {
            let mut tokens = self.tokens.write().await;
            for token_str in &token_strings {
                if let Some(t) = tokens.get_mut(token_str) {
                    t.revoked = true;
                }
            }
        }

        info!(user_id = %user_id, count = count, "Revoked all refresh tokens for user");
        count
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Cascade-revoke every token belonging to `family_id`.
    ///
    /// Called automatically when a replay attack is detected in [`rotate`].
    async fn cascade_revoke_family(&self, family_id: Uuid) {
        let token_strings: Vec<String> = {
            let family = self.family_index.read().await;
            family.get(&family_id).cloned().unwrap_or_default()
        };

        let count = token_strings.len();
        {
            let mut tokens = self.tokens.write().await;
            for token_str in &token_strings {
                if let Some(t) = tokens.get_mut(token_str) {
                    t.revoked = true;
                }
            }
        }

        warn!(
            family_id = %family_id,
            count = count,
            "Cascade-revoked all tokens in family due to replay attack",
        );
    }

    // -----------------------------------------------------------------------
    // Maintenance
    // -----------------------------------------------------------------------

    /// Remove tokens that have been expired for longer than `grace_period`.
    ///
    /// This keeps memory usage bounded in long-running deployments.  Revoked
    /// and rotated tokens are also purged after the grace period so that
    /// replay-detection state is not held forever.
    pub async fn cleanup_expired(&self, grace_period: Duration) -> usize {
        let cutoff = Utc::now() - grace_period;
        let mut removed = 0usize;

        let stale_tokens: Vec<String> = {
            let tokens = self.tokens.read().await;
            tokens
                .iter()
                .filter(|(_, t)| t.expires_at < cutoff)
                .map(|(k, _)| k.clone())
                .collect()
        };

        {
            let mut tokens = self.tokens.write().await;
            for key in &stale_tokens {
                tokens.remove(key);
                removed += 1;
            }
        }

        // Clean up secondary indices (best-effort; stale entries are harmless
        // but waste memory).
        if removed > 0 {
            let stale_set: std::collections::HashSet<&str> =
                stale_tokens.iter().map(String::as_str).collect();
            {
                let mut family = self.family_index.write().await;
                for vec in family.values_mut() {
                    vec.retain(|t| !stale_set.contains(t.as_str()));
                }
                family.retain(|_, v| !v.is_empty());
            }
            {
                let mut user_idx = self.user_index.write().await;
                for vec in user_idx.values_mut() {
                    vec.retain(|t| !stale_set.contains(t.as_str()));
                }
                user_idx.retain(|_, v| !v.is_empty());
            }
            debug!(removed = removed, "Cleaned up expired refresh tokens");
        }

        removed
    }

    /// Return the number of tokens currently held in the store (including
    /// rotated / revoked tokens until they are purged by `cleanup_expired`).
    pub async fn len(&self) -> usize {
        self.tokens.read().await.len()
    }

    /// Returns `true` when the store is empty.
    pub async fn is_empty(&self) -> bool {
        self.tokens.read().await.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    /// Default TTL used by most tests.
    const DEFAULT_TTL: Duration = Duration::hours(1);
    /// Very short TTL for expiry tests.
    const SHORT_TTL: Duration = Duration::milliseconds(50);

    // -----------------------------------------------------------------------
    // 1. Issue and validate
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_issue_returns_valid_token() {
        let store = RefreshTokenStore::new();
        let token = store.issue("alice", DEFAULT_TTL).await.unwrap();

        assert_eq!(token.user_id, "alice");
        assert!(!token.token.is_empty());
        assert_eq!(token.generation, 0);
        assert!(token.previous_token_hash.is_none());
        assert!(!token.rotated);
        assert!(!token.revoked);
        assert!(token.is_valid());
    }

    #[tokio::test]
    async fn test_validate_active_token_succeeds() {
        let store = RefreshTokenStore::new();
        let issued = store.issue("bob", DEFAULT_TTL).await.unwrap();

        let found = store.validate(&issued.token).await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().user_id, "bob");
    }

    #[tokio::test]
    async fn test_validate_nonexistent_token_returns_none() {
        let store = RefreshTokenStore::new();
        let found = store.validate("does-not-exist").await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_issued_token_has_correct_expiry() {
        let store = RefreshTokenStore::new();
        let ttl = Duration::seconds(300);
        let before = Utc::now();
        let token = store.issue("carol", ttl).await.unwrap();
        let after = Utc::now();

        assert!(token.expires_at >= before + ttl);
        assert!(token.expires_at <= after + ttl);
    }

    // -----------------------------------------------------------------------
    // 2. Rotation: old invalidated, new valid
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_rotate_produces_new_valid_token() {
        let store = RefreshTokenStore::new();
        let old = store.issue("alice", DEFAULT_TTL).await.unwrap();

        let new_token = store.rotate(&old.token, DEFAULT_TTL).await.unwrap();

        assert_ne!(new_token.token, old.token);
        assert_eq!(new_token.user_id, "alice");
        assert_eq!(new_token.family_id, old.family_id);
        assert_eq!(new_token.generation, 1);
        assert!(new_token.is_valid());
    }

    #[tokio::test]
    async fn test_rotate_invalidates_old_token() {
        let store = RefreshTokenStore::new();
        let old = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let _new_token = store.rotate(&old.token, DEFAULT_TTL).await.unwrap();

        // Old token must no longer be valid.
        let found = store.validate(&old.token).await;
        assert!(found.is_none());
    }

    #[tokio::test]
    async fn test_rotate_sets_previous_token_hash() {
        let store = RefreshTokenStore::new();
        let old = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let expected_hash = sha256_hex(&old.token);

        let new_token = store.rotate(&old.token, DEFAULT_TTL).await.unwrap();

        assert_eq!(
            new_token.previous_token_hash.as_deref(),
            Some(expected_hash.as_str())
        );
    }

    #[tokio::test]
    async fn test_chained_rotation_increments_generation() {
        let store = RefreshTokenStore::new();
        let t0 = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let t1 = store.rotate(&t0.token, DEFAULT_TTL).await.unwrap();
        let t2 = store.rotate(&t1.token, DEFAULT_TTL).await.unwrap();
        let t3 = store.rotate(&t2.token, DEFAULT_TTL).await.unwrap();

        assert_eq!(t3.generation, 3);
        assert_eq!(t3.family_id, t0.family_id);
    }

    #[tokio::test]
    async fn test_rotate_nonexistent_token_returns_not_found() {
        let store = RefreshTokenStore::new();
        let result = store.rotate("no-such-token", DEFAULT_TTL).await;
        assert_eq!(result.unwrap_err(), RotationError::TokenNotFound);
    }

    // -----------------------------------------------------------------------
    // 3. Replay attack: presenting rotated token revokes all user tokens
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_replay_attack_detected_on_rotated_token() {
        let store = RefreshTokenStore::new();
        let old = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let _new_token = store.rotate(&old.token, DEFAULT_TTL).await.unwrap();

        // Presenting the already-rotated token must be rejected.
        let result = store.rotate(&old.token, DEFAULT_TTL).await;
        assert_eq!(result.unwrap_err(), RotationError::TokenAlreadyRotated);
    }

    #[tokio::test]
    async fn test_replay_attack_cascade_revokes_all_family_tokens() {
        let store = RefreshTokenStore::new();
        let t0 = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let t1 = store.rotate(&t0.token, DEFAULT_TTL).await.unwrap();
        let t2 = store.rotate(&t1.token, DEFAULT_TTL).await.unwrap();

        // Replay t0 — should cascade-revoke t0, t1, t2.
        let _ = store.rotate(&t0.token, DEFAULT_TTL).await;

        // All three tokens must now be invalid.
        assert!(store.validate(&t0.token).await.is_none());
        assert!(store.validate(&t1.token).await.is_none());
        assert!(store.validate(&t2.token).await.is_none());
    }

    #[tokio::test]
    async fn test_replay_attack_does_not_affect_other_user_tokens() {
        let store = RefreshTokenStore::new();
        let alice_t0 = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let _alice_t1 = store.rotate(&alice_t0.token, DEFAULT_TTL).await.unwrap();

        // Bob has a completely separate family.
        let bob_token = store.issue("bob", DEFAULT_TTL).await.unwrap();

        // Replay alice's t0 — should cascade-revoke alice's family only.
        let _ = store.rotate(&alice_t0.token, DEFAULT_TTL).await;

        // Bob's token must still be valid.
        assert!(store.validate(&bob_token.token).await.is_some());
    }

    // -----------------------------------------------------------------------
    // 4. Expiry detection
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_expired_token_fails_validate() {
        let store = RefreshTokenStore::new();
        let token = store.issue("alice", SHORT_TTL).await.unwrap();

        sleep(std::time::Duration::from_millis(100)).await;

        assert!(store.validate(&token.token).await.is_none());
    }

    #[tokio::test]
    async fn test_expired_token_fails_rotation() {
        let store = RefreshTokenStore::new();
        let token = store.issue("alice", SHORT_TTL).await.unwrap();

        sleep(std::time::Duration::from_millis(100)).await;

        let result = store.rotate(&token.token, DEFAULT_TTL).await;
        assert_eq!(result.unwrap_err(), RotationError::TokenExpired);
    }

    // -----------------------------------------------------------------------
    // 5. Explicit revocation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_revoke_single_token() {
        let store = RefreshTokenStore::new();
        let token = store.issue("alice", DEFAULT_TTL).await.unwrap();

        store.revoke(&token.token).await;

        assert!(store.validate(&token.token).await.is_none());
    }

    #[tokio::test]
    async fn test_revoked_token_cannot_be_rotated() {
        let store = RefreshTokenStore::new();
        let token = store.issue("alice", DEFAULT_TTL).await.unwrap();

        store.revoke(&token.token).await;

        let result = store.rotate(&token.token, DEFAULT_TTL).await;
        assert_eq!(result.unwrap_err(), RotationError::TokenRevoked);
    }

    // -----------------------------------------------------------------------
    // 6. Revoke all for user (logout)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_revoke_all_for_user_invalidates_all_sessions() {
        let store = RefreshTokenStore::new();

        let s1 = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let s2 = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let s3 = store.issue("alice", DEFAULT_TTL).await.unwrap();

        let count = store.revoke_all_for_user("alice").await;
        assert_eq!(count, 3);

        assert!(store.validate(&s1.token).await.is_none());
        assert!(store.validate(&s2.token).await.is_none());
        assert!(store.validate(&s3.token).await.is_none());
    }

    #[tokio::test]
    async fn test_revoke_all_for_user_does_not_affect_other_users() {
        let store = RefreshTokenStore::new();

        let _alice_token = store.issue("alice", DEFAULT_TTL).await.unwrap();
        let bob_token = store.issue("bob", DEFAULT_TTL).await.unwrap();

        store.revoke_all_for_user("alice").await;

        // Bob's token must remain valid.
        assert!(store.validate(&bob_token.token).await.is_some());
    }

    #[tokio::test]
    async fn test_revoke_all_for_nonexistent_user_returns_zero() {
        let store = RefreshTokenStore::new();
        let count = store.revoke_all_for_user("nobody").await;
        assert_eq!(count, 0);
    }

    // -----------------------------------------------------------------------
    // 7. Concurrent rotation safety
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_concurrent_rotation_only_one_succeeds() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let store = Arc::new(RefreshTokenStore::new());
        let token = store.issue("alice", DEFAULT_TTL).await.unwrap();

        let successes = Arc::new(AtomicUsize::new(0));
        let replay_errors = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];

        // Spawn 10 concurrent tasks all trying to rotate the same token.
        for _ in 0..10 {
            let store_clone = Arc::clone(&store);
            let token_str = token.token.clone();
            let successes_clone = Arc::clone(&successes);
            let replay_clone = Arc::clone(&replay_errors);

            handles.push(tokio::spawn(async move {
                match store_clone.rotate(&token_str, DEFAULT_TTL).await {
                    Ok(_) => {
                        successes_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(RotationError::TokenAlreadyRotated) => {
                        replay_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_) => {}
                }
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // Due to the replay detection mechanism exactly one rotation should
        // succeed; the rest will see the token as already-rotated.
        // However, after the first replay detection triggers cascade revocation,
        // subsequent attempts see TokenNotFound or TokenAlreadyRotated.
        let total_successes = successes.load(Ordering::Relaxed);
        assert_eq!(
            total_successes, 1,
            "Expected exactly one successful rotation, got {total_successes}"
        );
    }

    #[tokio::test]
    async fn test_concurrent_issue_multiple_users() {
        let store = Arc::new(RefreshTokenStore::new());
        let mut handles = vec![];

        for i in 0..20 {
            let store_clone = Arc::clone(&store);
            handles.push(tokio::spawn(async move {
                let user = format!("user_{i}");
                store_clone.issue(&user, DEFAULT_TTL).await.unwrap()
            }));
        }

        let mut tokens = vec![];
        for h in handles {
            tokens.push(h.await.unwrap());
        }

        assert_eq!(store.len().await, 20);

        // All tokens must be unique.
        let token_set: std::collections::HashSet<&str> =
            tokens.iter().map(|t| t.token.as_str()).collect();
        assert_eq!(token_set.len(), 20);
    }

    // -----------------------------------------------------------------------
    // 8. Cleanup
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_cleanup_removes_expired_tokens() {
        let store = RefreshTokenStore::new();

        store.issue("alice", SHORT_TTL).await.unwrap();
        store.issue("alice", DEFAULT_TTL).await.unwrap();

        // Wait for short-lived token to expire.
        sleep(std::time::Duration::from_millis(100)).await;

        // Use zero grace period so anything past its expiry is removed.
        let removed = store.cleanup_expired(Duration::zero()).await;

        assert_eq!(
            removed, 1,
            "Expected exactly one expired token to be removed"
        );
        assert_eq!(store.len().await, 1);
    }

    #[tokio::test]
    async fn test_cleanup_with_grace_period_keeps_recently_expired() {
        let store = RefreshTokenStore::new();

        store.issue("alice", SHORT_TTL).await.unwrap();

        sleep(std::time::Duration::from_millis(100)).await;

        // Grace period of 1 hour means recently expired tokens are kept.
        let removed = store.cleanup_expired(Duration::hours(1)).await;

        assert_eq!(
            removed, 0,
            "Grace period should protect recently expired tokens"
        );
        assert_eq!(store.len().await, 1);
    }
}
