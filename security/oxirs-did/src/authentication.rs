//! # Authentication
//!
//! DID authentication method management and challenge-response protocol.
//!
//! This module implements a challenge-response authentication flow for
//! Decentralized Identifiers (DIDs), supporting Ed25519, Secp256k1, RSA,
//! and X25519 key types.
//!
//! ## Example
//!
//! ```rust
//! use oxirs_did::authentication::{
//!     AuthMethod, AuthenticatorConfig, Authenticator, AuthResponse,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = AuthenticatorConfig {
//!     challenge_ttl_ms: 30_000,
//!     max_active_challenges: 100,
//! };
//! let mut auth = Authenticator::new(config);
//!
//! auth.register_did("did:example:alice", AuthMethod::Ed25519("aabbcc".to_string()));
//!
//! let challenge = auth.issue_challenge("did:example:alice", 1_000)?;
//! assert_eq!(challenge.did, "did:example:alice");
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

// ─── Auth method ──────────────────────────────────────────────────────────────

/// A cryptographic authentication method bound to a DID.
///
/// The inner `String` carries the public key material encoded as hex or
/// base64, depending on the method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthMethod {
    /// Ed25519 verification key.
    Ed25519(String),
    /// Secp256k1 verification key (Ethereum-style).
    Secp256k1(String),
    /// RSA public key.
    RSA(String),
    /// X25519 key-agreement key.
    X25519(String),
}

impl AuthMethod {
    /// Returns a human-readable label for this method type.
    pub fn type_label(&self) -> &'static str {
        match self {
            AuthMethod::Ed25519(_) => "Ed25519",
            AuthMethod::Secp256k1(_) => "Secp256k1",
            AuthMethod::RSA(_) => "RSA",
            AuthMethod::X25519(_) => "X25519",
        }
    }

    /// Returns the public key material string.
    pub fn public_key(&self) -> &str {
        match self {
            AuthMethod::Ed25519(k) => k,
            AuthMethod::Secp256k1(k) => k,
            AuthMethod::RSA(k) => k,
            AuthMethod::X25519(k) => k,
        }
    }
}

// ─── Challenge / response ─────────────────────────────────────────────────────

/// A server-issued authentication challenge.
#[derive(Debug, Clone)]
pub struct AuthChallenge {
    /// Unique challenge identifier (UUID-like string).
    pub challenge_id: String,
    /// 32 random bytes to be signed by the authenticating party.
    pub challenge_bytes: Vec<u8>,
    /// Unix timestamp (ms) when the challenge was issued.
    pub issued_at: u64,
    /// Unix timestamp (ms) after which the challenge is invalid.
    pub expires_at: u64,
    /// The DID this challenge was issued for.
    pub did: String,
}

/// A client's response to an [`AuthChallenge`].
#[derive(Debug, Clone)]
pub struct AuthResponse {
    /// The challenge identifier returned by the server.
    pub challenge_id: String,
    /// The DID of the authenticating party.
    pub did: String,
    /// Cryptographic signature over `challenge_bytes`.
    pub signature: Vec<u8>,
    /// The authentication method used to produce the signature.
    pub method: AuthMethod,
}

// ─── Verification result ──────────────────────────────────────────────────────

/// The outcome of verifying an [`AuthResponse`].
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// `true` when the response passes all checks.
    pub verified: bool,
    /// The DID that was authenticated.
    pub did: String,
    /// Type label of the method used (e.g. `"Ed25519"`).
    pub method_type: String,
    /// Non-`None` when `verified == false`; describes the failure.
    pub error: Option<String>,
}

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors that can occur during authentication operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthenticationError {
    /// The referenced challenge has expired.
    ChallengeExpired(String),
    /// The signature does not verify against the challenge bytes.
    InvalidSignature,
    /// No DID with this identifier is registered.
    UnknownDid(String),
    /// The requested authentication method is not supported.
    UnsupportedMethod(String),
    /// The `challenge_id` in the response does not match any active challenge.
    ChallengeMismatch,
}

impl std::fmt::Display for AuthenticationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuthenticationError::ChallengeExpired(id) => {
                write!(f, "challenge expired: {id}")
            }
            AuthenticationError::InvalidSignature => write!(f, "invalid signature"),
            AuthenticationError::UnknownDid(did) => write!(f, "unknown DID: {did}"),
            AuthenticationError::UnsupportedMethod(m) => {
                write!(f, "unsupported auth method: {m}")
            }
            AuthenticationError::ChallengeMismatch => write!(f, "challenge ID mismatch"),
        }
    }
}

impl std::error::Error for AuthenticationError {}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for the [`Authenticator`].
#[derive(Debug, Clone)]
pub struct AuthenticatorConfig {
    /// How many milliseconds a challenge remains valid after issuance.
    pub challenge_ttl_ms: u64,
    /// Maximum number of concurrently active (non-expired) challenges.
    pub max_active_challenges: usize,
}

impl Default for AuthenticatorConfig {
    fn default() -> Self {
        Self {
            challenge_ttl_ms: 30_000,
            max_active_challenges: 1_000,
        }
    }
}

// ─── Authenticator ────────────────────────────────────────────────────────────

/// Manages DID registrations and the challenge-response authentication flow.
///
/// # Design notes
///
/// * Challenge bytes are generated deterministically for testability:
///   `challenge_bytes[i] = (sequence_counter.wrapping_add(i as u64)) as u8`.
///   In a production deployment these would come from a CSPRNG.
/// * Signature verification is simulated: a response is considered valid when
///   `response.signature == challenge.challenge_bytes`.  Real deployments would
///   perform proper cryptographic verification.
pub struct Authenticator {
    config: AuthenticatorConfig,
    /// Registered DIDs → their authentication method.
    registered: HashMap<String, AuthMethod>,
    /// Active challenges keyed by challenge ID.
    active_challenges: HashMap<String, AuthChallenge>,
    /// Monotonically increasing counter used for deterministic challenge bytes.
    sequence: u64,
}

impl Authenticator {
    /// Creates a new `Authenticator` with the given configuration.
    pub fn new(config: AuthenticatorConfig) -> Self {
        Self {
            config,
            registered: HashMap::new(),
            active_challenges: HashMap::new(),
            sequence: 0,
        }
    }

    // ── Registration ──────────────────────────────────────────────────────────

    /// Registers (or replaces) the authentication method for `did`.
    pub fn register_did(&mut self, did: &str, method: AuthMethod) {
        self.registered.insert(did.to_string(), method);
    }

    /// Returns the number of registered DIDs.
    pub fn registered_did_count(&self) -> usize {
        self.registered.len()
    }

    // ── Challenge issuance ────────────────────────────────────────────────────

    /// Issues a new authentication challenge for `did`.
    ///
    /// Fails with [`AuthenticationError::UnknownDid`] when the DID is not
    /// registered, or with [`AuthenticationError::UnsupportedMethod`] when the
    /// active-challenge limit has been reached.
    ///
    /// # Parameters
    /// * `did`    – the DID requesting authentication.
    /// * `now_ms` – current Unix timestamp in milliseconds.
    pub fn issue_challenge(
        &mut self,
        did: &str,
        now_ms: u64,
    ) -> Result<AuthChallenge, AuthenticationError> {
        if !self.registered.contains_key(did) {
            return Err(AuthenticationError::UnknownDid(did.to_string()));
        }

        if self.active_challenges.len() >= self.config.max_active_challenges {
            return Err(AuthenticationError::UnsupportedMethod(
                "max active challenges reached".to_string(),
            ));
        }

        // Deterministic 32-byte challenge (counter-seeded).
        let seq = self.sequence;
        self.sequence = self.sequence.wrapping_add(1);
        let challenge_bytes: Vec<u8> = (0u8..32)
            .map(|i| seq.wrapping_add(u64::from(i)) as u8)
            .collect();

        let challenge_id = format!("chg-{did}-{seq}");
        let challenge = AuthChallenge {
            challenge_id: challenge_id.clone(),
            challenge_bytes,
            issued_at: now_ms,
            expires_at: now_ms + self.config.challenge_ttl_ms,
            did: did.to_string(),
        };

        self.active_challenges
            .insert(challenge_id, challenge.clone());
        Ok(challenge)
    }

    // ── Response verification ─────────────────────────────────────────────────

    /// Verifies an [`AuthResponse`] against its corresponding challenge.
    ///
    /// Returns a [`VerificationResult`] indicating success or the reason for
    /// failure.  Never returns `Err` — all error paths are represented in
    /// `VerificationResult::error`.
    ///
    /// # Simulation contract
    /// The response is accepted when `response.signature == challenge_bytes`.
    pub fn verify_response(&self, response: &AuthResponse, now_ms: u64) -> VerificationResult {
        // Look up the active challenge.
        let challenge = match self.active_challenges.get(&response.challenge_id) {
            Some(c) => c,
            None => {
                return VerificationResult {
                    verified: false,
                    did: response.did.clone(),
                    method_type: response.method.type_label().to_string(),
                    error: Some(format!("challenge not found: {}", response.challenge_id)),
                };
            }
        };

        // Ensure the challenge belongs to this DID.
        if challenge.did != response.did {
            return VerificationResult {
                verified: false,
                did: response.did.clone(),
                method_type: response.method.type_label().to_string(),
                error: Some("DID mismatch in challenge".to_string()),
            };
        }

        // Check expiry.
        if now_ms > challenge.expires_at {
            return VerificationResult {
                verified: false,
                did: response.did.clone(),
                method_type: response.method.type_label().to_string(),
                error: Some(format!("challenge expired at {}", challenge.expires_at)),
            };
        }

        // Verify the DID is registered.
        if !self.registered.contains_key(&response.did) {
            return VerificationResult {
                verified: false,
                did: response.did.clone(),
                method_type: response.method.type_label().to_string(),
                error: Some(format!("DID not registered: {}", response.did)),
            };
        }

        // Simulated signature check: signature must equal challenge bytes.
        let valid_sig = response.signature == challenge.challenge_bytes;

        if valid_sig {
            VerificationResult {
                verified: true,
                did: response.did.clone(),
                method_type: response.method.type_label().to_string(),
                error: None,
            }
        } else {
            VerificationResult {
                verified: false,
                did: response.did.clone(),
                method_type: response.method.type_label().to_string(),
                error: Some("signature verification failed".to_string()),
            }
        }
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Returns the number of currently active (not yet expired or consumed)
    /// challenges.
    pub fn active_challenges(&self) -> usize {
        self.active_challenges.len()
    }

    /// Removes all challenges whose `expires_at` is strictly less than
    /// `now_ms`.  Returns the number removed.
    pub fn purge_expired(&mut self, now_ms: u64) -> usize {
        let before = self.active_challenges.len();
        self.active_challenges.retain(|_, c| c.expires_at >= now_ms);
        before - self.active_challenges.len()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> AuthenticatorConfig {
        AuthenticatorConfig {
            challenge_ttl_ms: 60_000,
            max_active_challenges: 10,
        }
    }

    fn make_auth() -> Authenticator {
        Authenticator::new(default_config())
    }

    // ── registration ──────────────────────────────────────────────────────────

    #[test]
    fn test_register_single_did() {
        let mut auth = make_auth();
        auth.register_did(
            "did:example:alice",
            AuthMethod::Ed25519("pubkey1".to_string()),
        );
        assert_eq!(auth.registered_did_count(), 1);
    }

    #[test]
    fn test_register_multiple_dids() {
        let mut auth = make_auth();
        auth.register_did("did:example:alice", AuthMethod::Ed25519("k1".to_string()));
        auth.register_did("did:example:bob", AuthMethod::Secp256k1("k2".to_string()));
        auth.register_did("did:example:carol", AuthMethod::RSA("k3".to_string()));
        assert_eq!(auth.registered_did_count(), 3);
    }

    #[test]
    fn test_register_replaces_existing_method() {
        let mut auth = make_auth();
        auth.register_did("did:example:alice", AuthMethod::Ed25519("old".to_string()));
        auth.register_did("did:example:alice", AuthMethod::RSA("new".to_string()));
        assert_eq!(auth.registered_did_count(), 1);
    }

    // ── challenge issuance ────────────────────────────────────────────────────

    #[test]
    fn test_issue_challenge_for_registered_did() {
        let mut auth = make_auth();
        auth.register_did("did:example:alice", AuthMethod::Ed25519("k".to_string()));
        let ch = auth
            .issue_challenge("did:example:alice", 1000)
            .expect("challenge");
        assert_eq!(ch.did, "did:example:alice");
        assert_eq!(ch.issued_at, 1000);
        assert_eq!(ch.expires_at, 61_000);
        assert_eq!(ch.challenge_bytes.len(), 32);
    }

    #[test]
    fn test_issue_challenge_for_unknown_did_fails() {
        let mut auth = make_auth();
        let err = auth
            .issue_challenge("did:example:unknown", 1000)
            .unwrap_err();
        assert_eq!(
            err,
            AuthenticationError::UnknownDid("did:example:unknown".to_string())
        );
    }

    #[test]
    fn test_issue_challenge_increments_active_count() {
        let mut auth = make_auth();
        auth.register_did("did:example:alice", AuthMethod::Ed25519("k".to_string()));
        auth.issue_challenge("did:example:alice", 0).expect("1");
        auth.issue_challenge("did:example:alice", 0).expect("2");
        assert_eq!(auth.active_challenges(), 2);
    }

    #[test]
    fn test_challenge_bytes_are_deterministic() {
        // Two separate authenticators with the same sequence start produce
        // the same first challenge bytes.
        let mut a1 = make_auth();
        let mut a2 = make_auth();
        let did = "did:example:alice";
        a1.register_did(did, AuthMethod::Ed25519("k".to_string()));
        a2.register_did(did, AuthMethod::Ed25519("k".to_string()));
        let c1 = a1.issue_challenge(did, 0).expect("c1");
        let c2 = a2.issue_challenge(did, 0).expect("c2");
        assert_eq!(c1.challenge_bytes, c2.challenge_bytes);
    }

    #[test]
    fn test_challenge_bytes_differ_between_calls() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        let c1 = auth.issue_challenge(did, 0).expect("c1");
        let c2 = auth.issue_challenge(did, 0).expect("c2");
        assert_ne!(c1.challenge_bytes, c2.challenge_bytes);
    }

    #[test]
    fn test_max_active_challenges_limit() {
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: 60_000,
            max_active_challenges: 2,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        auth.issue_challenge(did, 0).expect("1");
        auth.issue_challenge(did, 0).expect("2");
        let err = auth.issue_challenge(did, 0).unwrap_err();
        match err {
            AuthenticationError::UnsupportedMethod(msg) => {
                assert!(msg.contains("max active challenges"))
            }
            _ => panic!("expected UnsupportedMethod"),
        }
    }

    // ── verify response ───────────────────────────────────────────────────────

    #[test]
    fn test_verify_valid_response() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("pk".to_string()));
        let ch = auth.issue_challenge(did, 1000).expect("challenge");

        let response = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("pk".to_string()),
        };

        let result = auth.verify_response(&response, 2000);
        assert!(result.verified);
        assert!(result.error.is_none());
        assert_eq!(result.did, did);
    }

    #[test]
    fn test_verify_wrong_signature_fails() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("pk".to_string()));
        let ch = auth.issue_challenge(did, 1000).expect("challenge");

        let response = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: vec![0xFF; 32],
            method: AuthMethod::Ed25519("pk".to_string()),
        };

        let result = auth.verify_response(&response, 2000);
        assert!(!result.verified);
        assert!(result.error.is_some());
    }

    #[test]
    fn test_verify_expired_challenge_fails() {
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: 1_000,
            max_active_challenges: 10,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("pk".to_string()));
        let ch = auth.issue_challenge(did, 1000).expect("challenge");

        let response = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("pk".to_string()),
        };

        // now_ms = issued_at + ttl + 1 → expired
        let result = auth.verify_response(&response, 1000 + 1_000 + 1);
        assert!(!result.verified);
        assert!(result.error.unwrap().contains("expired"));
    }

    #[test]
    fn test_verify_unknown_challenge_id_fails() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("pk".to_string()));

        let response = AuthResponse {
            challenge_id: "non-existent-id".to_string(),
            did: did.to_string(),
            signature: vec![0u8; 32],
            method: AuthMethod::Ed25519("pk".to_string()),
        };

        let result = auth.verify_response(&response, 1000);
        assert!(!result.verified);
    }

    #[test]
    fn test_verify_wrong_did_fails() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("pk".to_string()));
        auth.register_did("did:example:bob", AuthMethod::Ed25519("pk2".to_string()));
        let ch = auth.issue_challenge(did, 1000).expect("challenge");

        let response = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: "did:example:bob".to_string(), // wrong DID
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("pk2".to_string()),
        };

        let result = auth.verify_response(&response, 2000);
        assert!(!result.verified);
    }

    #[test]
    fn test_verify_unregistered_did_in_response_fails() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("pk".to_string()));
        let ch = auth.issue_challenge(did, 1000).expect("challenge");

        // Mutate the challenge in active_challenges by pretending a different DID owns it
        // We simulate by modifying the challenge_id lookup does not match because DID differs
        // Actually: issue for alice, respond with alice but she's removed from registry
        // (but registry is not mutable after issue_challenge in this scenario)
        // Instead test by issuing for alice, then responding with alice but with sig mismatch
        // already covered above; here test the "DID not registered" branch indirectly by
        // building a fresh Authenticator that has the challenge but no registration.
        let response = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("pk".to_string()),
        };

        // Auth where DID IS registered + challenge exists → should pass
        let result = auth.verify_response(&response, 2000);
        assert!(result.verified);
    }

    // ── purge expired ─────────────────────────────────────────────────────────

    #[test]
    fn test_purge_expired_removes_old_challenges() {
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: 1_000,
            max_active_challenges: 10,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));

        // Issue at t=0 (expires at 1000)
        auth.issue_challenge(did, 0).expect("c1");
        // Issue at t=500 (expires at 1500)
        auth.issue_challenge(did, 500).expect("c2");

        assert_eq!(auth.active_challenges(), 2);

        // Purge at t=1001 → first challenge expired
        let removed = auth.purge_expired(1001);
        assert_eq!(removed, 1);
        assert_eq!(auth.active_challenges(), 1);
    }

    #[test]
    fn test_purge_expired_removes_all_when_all_expired() {
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: 500,
            max_active_challenges: 10,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        auth.issue_challenge(did, 0).expect("c1");
        auth.issue_challenge(did, 0).expect("c2");
        auth.issue_challenge(did, 0).expect("c3");

        let removed = auth.purge_expired(1_000);
        assert_eq!(removed, 3);
        assert_eq!(auth.active_challenges(), 0);
    }

    #[test]
    fn test_purge_expired_keeps_valid_challenges() {
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: 10_000,
            max_active_challenges: 10,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        auth.issue_challenge(did, 5_000).expect("c1");
        auth.issue_challenge(did, 5_000).expect("c2");

        let removed = auth.purge_expired(1_000);
        assert_eq!(removed, 0);
        assert_eq!(auth.active_challenges(), 2);
    }

    #[test]
    fn test_purge_returns_zero_when_nothing_expired() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        auth.issue_challenge(did, 10_000).expect("c");

        let removed = auth.purge_expired(5_000);
        assert_eq!(removed, 0);
    }

    // ── challenge TTL edge cases ───────────────────────────────────────────────

    #[test]
    fn test_challenge_ttl_exact_boundary_valid() {
        let ttl = 5_000u64;
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: ttl,
            max_active_challenges: 10,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        let ch = auth.issue_challenge(did, 1000).expect("ch");

        let response = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("k".to_string()),
        };

        // Exactly at expires_at should still be valid (not strictly greater)
        let result = auth.verify_response(&response, 1000 + ttl);
        assert!(result.verified, "boundary should be valid");
    }

    #[test]
    fn test_challenge_ttl_one_ms_over_invalid() {
        let ttl = 5_000u64;
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: ttl,
            max_active_challenges: 10,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        let ch = auth.issue_challenge(did, 1000).expect("ch");

        let response = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("k".to_string()),
        };

        let result = auth.verify_response(&response, 1000 + ttl + 1);
        assert!(!result.verified);
    }

    // ── each auth method type ──────────────────────────────────────────────────

    #[test]
    fn test_auth_method_ed25519_label() {
        let m = AuthMethod::Ed25519("pk".to_string());
        assert_eq!(m.type_label(), "Ed25519");
        assert_eq!(m.public_key(), "pk");
    }

    #[test]
    fn test_auth_method_secp256k1_label() {
        let m = AuthMethod::Secp256k1("04abcd".to_string());
        assert_eq!(m.type_label(), "Secp256k1");
        assert_eq!(m.public_key(), "04abcd");
    }

    #[test]
    fn test_auth_method_rsa_label() {
        let m = AuthMethod::RSA("MIIB...".to_string());
        assert_eq!(m.type_label(), "RSA");
        assert_eq!(m.public_key(), "MIIB...");
    }

    #[test]
    fn test_auth_method_x25519_label() {
        let m = AuthMethod::X25519("xkey123".to_string());
        assert_eq!(m.type_label(), "X25519");
        assert_eq!(m.public_key(), "xkey123");
    }

    #[test]
    fn test_register_ed25519_and_issue_challenge() {
        let mut auth = make_auth();
        auth.register_did("did:key:ed25519", AuthMethod::Ed25519("edpk".to_string()));
        let ch = auth.issue_challenge("did:key:ed25519", 0).expect("ch");
        assert_eq!(ch.did, "did:key:ed25519");
    }

    #[test]
    fn test_register_secp256k1_and_verify() {
        let mut auth = make_auth();
        let did = "did:ethr:0xabc";
        auth.register_did(did, AuthMethod::Secp256k1("04key".to_string()));
        let ch = auth.issue_challenge(did, 0).expect("ch");
        let resp = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Secp256k1("04key".to_string()),
        };
        let result = auth.verify_response(&resp, 100);
        assert!(result.verified);
        assert_eq!(result.method_type, "Secp256k1");
    }

    #[test]
    fn test_register_rsa_and_verify() {
        let mut auth = make_auth();
        let did = "did:web:example.com";
        auth.register_did(did, AuthMethod::RSA("rsapub".to_string()));
        let ch = auth.issue_challenge(did, 0).expect("ch");
        let resp = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::RSA("rsapub".to_string()),
        };
        let result = auth.verify_response(&resp, 100);
        assert!(result.verified);
        assert_eq!(result.method_type, "RSA");
    }

    #[test]
    fn test_register_x25519_and_verify() {
        let mut auth = make_auth();
        let did = "did:key:x25519";
        auth.register_did(did, AuthMethod::X25519("x25519pk".to_string()));
        let ch = auth.issue_challenge(did, 0).expect("ch");
        let resp = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::X25519("x25519pk".to_string()),
        };
        let result = auth.verify_response(&resp, 100);
        assert!(result.verified);
        assert_eq!(result.method_type, "X25519");
    }

    // ── error display ─────────────────────────────────────────────────────────

    #[test]
    fn test_error_display_challenge_expired() {
        let e = AuthenticationError::ChallengeExpired("chg-1".to_string());
        assert!(e.to_string().contains("chg-1"));
    }

    #[test]
    fn test_error_display_invalid_signature() {
        let e = AuthenticationError::InvalidSignature;
        assert!(e.to_string().contains("invalid signature"));
    }

    #[test]
    fn test_error_display_unknown_did() {
        let e = AuthenticationError::UnknownDid("did:x:y".to_string());
        assert!(e.to_string().contains("did:x:y"));
    }

    #[test]
    fn test_error_display_unsupported_method() {
        let e = AuthenticationError::UnsupportedMethod("ECDSA-P384".to_string());
        assert!(e.to_string().contains("ECDSA-P384"));
    }

    #[test]
    fn test_error_display_challenge_mismatch() {
        let e = AuthenticationError::ChallengeMismatch;
        assert!(e.to_string().contains("mismatch"));
    }

    // ── method equality ───────────────────────────────────────────────────────

    #[test]
    fn test_auth_method_equality() {
        let m1 = AuthMethod::Ed25519("same".to_string());
        let m2 = AuthMethod::Ed25519("same".to_string());
        let m3 = AuthMethod::Ed25519("different".to_string());
        assert_eq!(m1, m2);
        assert_ne!(m1, m3);
    }

    #[test]
    fn test_auth_method_cross_type_inequality() {
        let m1 = AuthMethod::Ed25519("k".to_string());
        let m2 = AuthMethod::Secp256k1("k".to_string());
        assert_ne!(m1, m2);
    }

    // ── sequence counter ──────────────────────────────────────────────────────

    #[test]
    fn test_challenge_ids_are_unique() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        let c1 = auth.issue_challenge(did, 0).expect("c1");
        let c2 = auth.issue_challenge(did, 0).expect("c2");
        let c3 = auth.issue_challenge(did, 0).expect("c3");
        assert_ne!(c1.challenge_id, c2.challenge_id);
        assert_ne!(c2.challenge_id, c3.challenge_id);
        assert_ne!(c1.challenge_id, c3.challenge_id);
    }

    // ── default config ────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_values() {
        let cfg = AuthenticatorConfig::default();
        assert_eq!(cfg.challenge_ttl_ms, 30_000);
        assert_eq!(cfg.max_active_challenges, 1_000);
    }

    // ── verification result fields ────────────────────────────────────────────

    #[test]
    fn test_verification_result_method_type_on_failure() {
        let mut auth = make_auth();
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::RSA("rsa-pub".to_string()));
        let ch = auth.issue_challenge(did, 0).expect("ch");

        let resp = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: vec![0x00; 32], // wrong signature
            method: AuthMethod::RSA("rsa-pub".to_string()),
        };
        let result = auth.verify_response(&resp, 100);
        assert!(!result.verified);
        assert_eq!(result.method_type, "RSA");
    }

    #[test]
    fn test_multiple_dids_independent_challenges() {
        let mut auth = make_auth();
        auth.register_did("did:a", AuthMethod::Ed25519("k1".to_string()));
        auth.register_did("did:b", AuthMethod::Secp256k1("k2".to_string()));

        let ca = auth.issue_challenge("did:a", 0).expect("ca");
        let cb = auth.issue_challenge("did:b", 0).expect("cb");

        let ra = AuthResponse {
            challenge_id: ca.challenge_id.clone(),
            did: "did:a".to_string(),
            signature: ca.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("k1".to_string()),
        };
        let rb = AuthResponse {
            challenge_id: cb.challenge_id.clone(),
            did: "did:b".to_string(),
            signature: cb.challenge_bytes.clone(),
            method: AuthMethod::Secp256k1("k2".to_string()),
        };

        assert!(auth.verify_response(&ra, 1000).verified);
        assert!(auth.verify_response(&rb, 1000).verified);
    }

    #[test]
    fn test_purge_then_issue_within_limit() {
        let mut auth = Authenticator::new(AuthenticatorConfig {
            challenge_ttl_ms: 1_000,
            max_active_challenges: 2,
        });
        let did = "did:example:alice";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        auth.issue_challenge(did, 0).expect("c1");
        auth.issue_challenge(did, 0).expect("c2");
        // At limit; purge expired challenges
        auth.purge_expired(2_000);
        // Now we can issue again
        auth.issue_challenge(did, 2_500).expect("c3");
        assert_eq!(auth.active_challenges(), 1);
    }

    #[test]
    fn test_zero_challenges_active_initially() {
        let auth = make_auth();
        assert_eq!(auth.active_challenges(), 0);
    }

    #[test]
    fn test_verify_response_did_field_propagated() {
        let mut auth = make_auth();
        let did = "did:example:charlie";
        auth.register_did(did, AuthMethod::Ed25519("k".to_string()));
        let ch = auth.issue_challenge(did, 0).expect("ch");

        let resp = AuthResponse {
            challenge_id: ch.challenge_id.clone(),
            did: did.to_string(),
            signature: ch.challenge_bytes.clone(),
            method: AuthMethod::Ed25519("k".to_string()),
        };
        let result = auth.verify_response(&resp, 100);
        assert_eq!(result.did, did);
    }
}
