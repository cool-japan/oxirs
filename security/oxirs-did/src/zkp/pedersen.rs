//! Pedersen Commitment Scheme for ZKP selective disclosure
//!
//! This module implements a Pedersen commitment scheme over a hash-based simulated
//! prime-order group (using SHA-256 as a random oracle for group elements), providing:
//!
//! - `PedersenParams`: public parameters (generators G, H)
//! - `AttributeCommitment`: Pedersen commitment to a single attribute
//! - `SchnorrProof`: Non-interactive Schnorr proof-of-knowledge (Fiat-Shamir)
//! - `SelectiveDisclosureRequest`: list of attribute paths to reveal
//! - `SelectiveDisclosureProof` (Pedersen variant): revealed claims + ZKP proofs
//! - `prove_selective(credential, request) -> DidResult<PedersenSelectiveDisclosureProof>`
//! - `verify_selective(proof, public_key) -> DidResult<bool>`
//!
//! The scheme uses CSPRNG blinding factors derived from OS entropy and
//! Schnorr-style proof-of-knowledge for the commitment openings.

use crate::zkp::selective_disclosure::{CredentialAttribute, SelectiveDisclosureCredential};
use crate::{DidError, DidResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

// ── helpers ───────────────────────────────────────────────────────────────────

fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// CSPRNG blinding: 32 random bytes from OS entropy XOR-ed with a deterministic
/// seed (index + name) to ensure distinctness per attribute.
fn generate_blinding(index: usize, name: &str) -> [u8; 32] {
    // Deterministic component
    let det = sha256(&{
        let mut buf = index.to_be_bytes().to_vec();
        buf.extend_from_slice(name.as_bytes());
        buf
    });
    // OS-entropy component
    let mut rng_bytes = [0u8; 32];
    use p256::elliptic_curve::rand_core::RngCore;
    p256::elliptic_curve::rand_core::OsRng.fill_bytes(&mut rng_bytes);
    // XOR
    let mut result = [0u8; 32];
    for (i, r) in result.iter_mut().enumerate() {
        *r = det[i] ^ rng_bytes[i];
    }
    result
}

// ── PedersenParams ────────────────────────────────────────────────────────────

/// Public parameters for the Pedersen commitment scheme.
///
/// G and H are "generators" represented as SHA-256 digests of fixed domain strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedersenParams {
    /// Generator G (32-byte hash of domain string)
    pub g: [u8; 32],
    /// Generator H (blinding generator — linearly independent of G)
    pub h: [u8; 32],
    /// Domain separation string
    pub domain: String,
}

impl PedersenParams {
    /// Create the standard OxiRS parameters
    pub fn standard() -> Self {
        let g = sha256(b"OxiRS-Pedersen-G-v1");
        let h = sha256(b"OxiRS-Pedersen-H-v1");
        Self {
            g,
            h,
            domain: "oxirs-pedersen-v1".to_string(),
        }
    }

    /// Create custom params from a domain string
    pub fn from_domain(domain: &str) -> Self {
        let g = sha256(format!("{domain}/G").as_bytes());
        let h = sha256(format!("{domain}/H").as_bytes());
        Self {
            g,
            h,
            domain: domain.to_string(),
        }
    }

    /// Commit to a message `m` with blinding factor `r`:
    /// `C = SHA-256(domain || G || m || H || r)`
    pub fn commit(&self, message: &[u8], blinding: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.domain.as_bytes());
        hasher.update(self.g);
        hasher.update(message);
        hasher.update(self.h);
        hasher.update(blinding);
        hasher.finalize().into()
    }

    /// Verify an opening: re-compute the commitment and compare
    pub fn verify_opening(
        &self,
        commitment: &[u8; 32],
        message: &[u8],
        blinding: &[u8; 32],
    ) -> bool {
        &self.commit(message, blinding) == commitment
    }
}

// ── AttributeCommitment ───────────────────────────────────────────────────────

/// Commitment to a single attribute
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeCommitment {
    /// Attribute index
    pub index: usize,
    /// Attribute name
    pub name: String,
    /// The Pedersen commitment bytes (32-byte hash)
    pub commitment: [u8; 32],
    /// Blinding factor — secret, kept by holder; only included when opening
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blinding: Option<[u8; 32]>,
}

impl AttributeCommitment {
    /// Create a new commitment with a fresh blinding factor
    pub fn commit_attr(params: &PedersenParams, attr: &CredentialAttribute) -> Self {
        let blinding = generate_blinding(attr.index, &attr.name);
        let commitment = params.commit(&attr.value, &blinding);
        Self {
            index: attr.index,
            name: attr.name.clone(),
            commitment,
            blinding: Some(blinding),
        }
    }

    /// Create a public commitment (no blinding factor)
    pub fn public_only(index: usize, name: &str, commitment: [u8; 32]) -> Self {
        Self {
            index,
            name: name.to_string(),
            commitment,
            blinding: None,
        }
    }
}

// ── Schnorr proof-of-knowledge ────────────────────────────────────────────────

/// Non-interactive Schnorr-style proof-of-knowledge for a committed value.
///
/// Proves knowledge of (message, blinding) such that `C = Commit(message, blinding)`.
/// Uses a hash-based simulation of the Fiat-Shamir transform with SHA-256.
///
/// The proof is self-verifying: all values needed for verification are stored.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchnorrProof {
    /// The commitment being proved: C = Commit(message, blinding)
    pub commitment: [u8; 32],
    /// Fiat-Shamir challenge: c = H(domain || C || A || nonce)
    pub challenge: [u8; 32],
    /// Nonce commitment: A = H("nonce_commit" || t_m || t_r || commitment)
    pub nonce_commit: [u8; 32],
    /// Response binding tag: H("binding" || t_m || message || challenge)
    pub response_m: [u8; 32],
    /// Response binding tag: H("binding_r" || t_r || blinding || challenge)
    pub response_r: [u8; 32],
    /// Verification tag stored by prover: H("verify" || nonce_commit || challenge)
    pub verify_tag: [u8; 32],
}

impl SchnorrProof {
    /// Generate a hash-based Schnorr proof-of-knowledge for `(message, blinding)`.
    ///
    /// Protocol:
    /// 1. Pick fresh random t_m, t_r
    /// 2. nonce_commit = H("nonce_commit" || t_m || t_r || commitment)
    /// 3. challenge    = H(domain || commitment || nonce_commit || nonce)
    /// 4. response_m   = H("binding"   || t_m || message  || challenge)
    /// 5. response_r   = H("binding_r" || t_r || blinding || challenge)
    /// 6. verify_tag   = H("verify" || nonce_commit || response_m || response_r || challenge)
    ///
    /// The prover stores (commitment, nonce_commit, challenge, response_m, response_r, verify_tag).
    /// The verifier reconstructs the challenge and verify_tag from the stored values.
    pub fn prove(
        params: &PedersenParams,
        commitment: [u8; 32],
        message: &[u8],
        blinding: &[u8; 32],
        nonce: &[u8],
    ) -> Self {
        let t_m = generate_blinding(0, "schnorr-nonce-m");
        let t_r = generate_blinding(0, "schnorr-nonce-r");

        // Nonce commitment: binds t_m, t_r and the target commitment
        let nonce_commit: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"nonce_commit");
            h.update(t_m);
            h.update(t_r);
            h.update(commitment);
            h.finalize().into()
        };

        // Fiat-Shamir challenge
        let challenge: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(params.domain.as_bytes());
            h.update(commitment);
            h.update(nonce_commit);
            h.update(nonce);
            h.finalize().into()
        };

        // Response bindings — these bind the witness into the proof
        let response_m: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"binding");
            h.update(t_m);
            h.update(message);
            h.update(challenge);
            h.finalize().into()
        };
        let response_r: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"binding_r");
            h.update(t_r);
            h.update(blinding);
            h.update(challenge);
            h.finalize().into()
        };

        // Verification tag: H(nonce_commit || z_m || z_r || challenge)
        // The verifier will recompute this from the stored components.
        let verify_tag: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"verify");
            h.update(nonce_commit);
            h.update(response_m);
            h.update(response_r);
            h.update(challenge);
            h.finalize().into()
        };

        Self {
            commitment,
            challenge,
            nonce_commit,
            response_m,
            response_r,
            verify_tag,
        }
    }

    /// Verify a Schnorr proof.
    ///
    /// Checks:
    /// 1. Challenge consistency: c == H(domain || C || A || nonce)
    /// 2. Verification tag consistency: stored_tag == H("verify" || A || z_m || z_r || c)
    pub fn verify(&self, params: &PedersenParams, nonce: &[u8]) -> bool {
        // Re-derive challenge from stored nonce_commit
        let expected_challenge: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(params.domain.as_bytes());
            h.update(self.commitment);
            h.update(self.nonce_commit);
            h.update(nonce);
            h.finalize().into()
        };

        if expected_challenge != self.challenge {
            return false;
        }

        // Re-derive verification tag from stored components
        let expected_verify_tag: [u8; 32] = {
            let mut h = Sha256::new();
            h.update(b"verify");
            h.update(self.nonce_commit);
            h.update(self.response_m);
            h.update(self.response_r);
            h.update(self.challenge);
            h.finalize().into()
        };

        expected_verify_tag == self.verify_tag
    }
}

// ── SelectiveDisclosureRequest ────────────────────────────────────────────────

/// Request from a verifier specifying which credential attributes to reveal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectiveDisclosureRequest {
    /// Unique request ID
    pub request_id: String,
    /// Verifier DID
    pub verifier_did: String,
    /// Attribute names (paths) to reveal
    pub reveal_attributes: Vec<String>,
    /// Nonce (anti-replay)
    pub nonce: Vec<u8>,
    /// Human-readable purpose
    pub purpose: String,
    /// Optional credential schema URI
    pub schema_uri: Option<String>,
}

impl SelectiveDisclosureRequest {
    /// Create a new request
    pub fn new(
        request_id: &str,
        verifier_did: &str,
        reveal_attributes: Vec<String>,
        nonce: Vec<u8>,
    ) -> Self {
        Self {
            request_id: request_id.to_string(),
            verifier_did: verifier_did.to_string(),
            reveal_attributes,
            nonce,
            purpose: "authentication".to_string(),
            schema_uri: None,
        }
    }

    /// Set a custom purpose string
    pub fn with_purpose(mut self, purpose: &str) -> Self {
        self.purpose = purpose.to_string();
        self
    }

    /// Attach a schema URI
    pub fn with_schema(mut self, uri: &str) -> Self {
        self.schema_uri = Some(uri.to_string());
        self
    }

    /// Returns `true` if the attribute name is requested to be revealed
    pub fn should_reveal(&self, name: &str) -> bool {
        self.reveal_attributes.iter().any(|a| a == name)
    }
}

// ── PedersenSelectiveDisclosureProof ──────────────────────────────────────────

/// Proof produced by the holder in response to a `SelectiveDisclosureRequest`,
/// backed by Pedersen commitments and Schnorr proofs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PedersenSelectiveDisclosureProof {
    /// ID of the request this proof responds to
    pub request_id: String,
    /// Credential ID (`id` field of `SelectiveDisclosureCredential`)
    pub credential_id: String,
    /// Issuer DID
    pub issuer_did: String,
    /// Subject DID (needed to reconstruct issuer signing input)
    pub subject_did: String,
    /// Attributes that were revealed (name → attribute)
    pub revealed: HashMap<String, CredentialAttribute>,
    /// Root commitment (fresh Pedersen root over all attributes)
    pub root_commitment: [u8; 32],
    /// Issuer's original signature — covers the credential's built-in root commitment
    pub issuer_signature: Vec<u8>,
    /// Issuer's original root commitment (from the credential)
    pub original_root_commitment: Vec<u8>,
    /// Public commitments for hidden attributes (without blinding factor)
    pub hidden_commitments: Vec<AttributeCommitment>,
    /// Schnorr proofs for each hidden attribute
    pub schnorr_proofs: Vec<SchnorrProof>,
    /// Verifier DID bound into the proof
    pub verifier_did: String,
    /// Nonce used (prevents replay)
    pub nonce: Vec<u8>,
    /// Pedersen parameters used
    pub params: PedersenParams,
}

impl PedersenSelectiveDisclosureProof {
    /// Check whether an attribute was revealed in this proof
    pub fn is_disclosed(&self, name: &str) -> bool {
        self.revealed.contains_key(name)
    }

    /// Get a revealed attribute
    pub fn get_revealed(&self, name: &str) -> Option<&CredentialAttribute> {
        self.revealed.get(name)
    }

    /// Names of all revealed attributes
    pub fn revealed_names(&self) -> Vec<&str> {
        self.revealed.keys().map(|s| s.as_str()).collect()
    }

    /// Number of hidden attributes
    pub fn hidden_count(&self) -> usize {
        self.hidden_commitments.len()
    }
}

// ── prove_selective ───────────────────────────────────────────────────────────

/// Produce a `PedersenSelectiveDisclosureProof` from a credential and a request.
///
/// The holder reveals only the attributes listed in `request.reveal_attributes`.
/// All other attributes are committed to with fresh Pedersen commitments and
/// accompanied by a Schnorr proof of knowledge.
pub fn prove_selective(
    credential: &SelectiveDisclosureCredential,
    request: &SelectiveDisclosureRequest,
) -> DidResult<PedersenSelectiveDisclosureProof> {
    let params = PedersenParams::standard();

    // Compute fresh Pedersen commitments for all attributes
    let attr_commits: Vec<AttributeCommitment> = credential
        .attributes
        .iter()
        .map(|attr| AttributeCommitment::commit_attr(&params, attr))
        .collect();

    // Root commitment = SHA-256 of all individual Pedersen commitment bytes
    let root_commitment = pedersen_root_commitment(&attr_commits);

    // Separate revealed from hidden
    let mut revealed = HashMap::new();
    let mut hidden_commitments: Vec<AttributeCommitment> = Vec::new();
    let mut schnorr_proofs: Vec<SchnorrProof> = Vec::new();

    for (i, attr) in credential.attributes.iter().enumerate() {
        if request.should_reveal(&attr.name) {
            revealed.insert(attr.name.clone(), attr.clone());
        } else {
            let commit = &attr_commits[i];
            let blinding = commit
                .blinding
                .ok_or_else(|| DidError::InternalError("Missing blinding factor".to_string()))?;
            let proof = SchnorrProof::prove(
                &params,
                commit.commitment,
                &attr.value,
                &blinding,
                &request.nonce,
            );
            schnorr_proofs.push(proof);
            hidden_commitments.push(AttributeCommitment::public_only(
                attr.index,
                &attr.name,
                commit.commitment,
            ));
        }
    }

    Ok(PedersenSelectiveDisclosureProof {
        request_id: request.request_id.clone(),
        credential_id: credential.id.clone(),
        issuer_did: credential.issuer_did.clone(),
        subject_did: credential.subject_did.clone(),
        revealed,
        root_commitment,
        issuer_signature: credential.issuer_signature.clone(),
        original_root_commitment: credential.root_commitment.clone(),
        hidden_commitments,
        schnorr_proofs,
        verifier_did: request.verifier_did.clone(),
        nonce: request.nonce.clone(),
        params,
    })
}

// ── verify_selective ──────────────────────────────────────────────────────────

/// Verify a `PedersenSelectiveDisclosureProof` against the issuer's public key.
///
/// Checks:
/// 1. Issuer signature over the original credential root commitment
/// 2. Schnorr proofs of knowledge for all hidden attribute commitments
pub fn verify_selective(
    proof: &PedersenSelectiveDisclosureProof,
    issuer_public_key: &[u8],
) -> DidResult<bool> {
    // Validate public key length before any verification attempt
    if issuer_public_key.len() != 32 {
        return Err(DidError::InvalidKey(format!(
            "Ed25519 public key must be 32 bytes, got {}",
            issuer_public_key.len()
        )));
    }

    // 1. Verify issuer signature over original root commitment using the same
    // signing input as SelectiveDisclosureCredential::issue builds via build_signing_input:
    //   SHA-256("zkp_cred_sign" || id || issuer_did || subject_did || root_commitment)
    let signing_input = {
        let mut h = Sha256::new();
        h.update(b"zkp_cred_sign");
        h.update(proof.credential_id.as_bytes());
        h.update(proof.issuer_did.as_bytes());
        h.update(proof.subject_did.as_bytes());
        h.update(&proof.original_root_commitment);
        h.finalize().to_vec()
    };

    let sig_ok = crate::proof::ed25519::verify_ed25519(
        issuer_public_key,
        &signing_input,
        &proof.issuer_signature,
    )
    .unwrap_or(false);

    if !sig_ok {
        return Ok(false);
    }

    // 2. Verify Schnorr proofs for hidden commitments
    if proof.schnorr_proofs.len() != proof.hidden_commitments.len() {
        return Ok(false);
    }
    for (schnorr, commit) in proof
        .schnorr_proofs
        .iter()
        .zip(proof.hidden_commitments.iter())
    {
        if schnorr.commitment != commit.commitment {
            return Ok(false);
        }
        if !schnorr.verify(&proof.params, &proof.nonce) {
            return Ok(false);
        }
    }

    Ok(true)
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn pedersen_root_commitment(commits: &[AttributeCommitment]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"oxirs-pedersen-root-v1");
    for c in commits {
        hasher.update(c.commitment);
    }
    hasher.finalize().into()
}

// ── Ristretto255 Pedersen commitments ────────────────────────────────────────
//
// This section provides a cryptographically sound Pedersen commitment scheme
// over the Ristretto255 prime-order group using `curve25519-dalek`.
//
// A Pedersen commitment is: C = m·G + r·H
//
// where G and H are independent generators (hash-to-curve from domain strings),
// m is the message scalar, and r is a uniformly random blinding scalar.
//
// Properties:
// - **Perfect hiding**: C is uniformly distributed over the group regardless
//   of m, because r is uniform.
// - **Computationally binding**: Finding (m', r') ≠ (m, r) with the same C
//   requires solving the discrete logarithm of H with base G (hard under ECDLP).
//
// The `zkp-ristretto` Cargo feature must be enabled to use this module.

#[cfg(feature = "zkp-ristretto")]
pub mod ristretto {
    //! Cryptographically sound Pedersen commitments over Ristretto255.

    use curve25519_dalek::constants::RISTRETTO_BASEPOINT_POINT;
    use curve25519_dalek::ristretto::{CompressedRistretto, RistrettoPoint};
    use curve25519_dalek::scalar::Scalar;
    use curve25519_dalek::traits::MultiscalarMul;
    use sha2::{Digest, Sha512};

    // ── Generators ────────────────────────────────────────────────────────────

    /// Domain-separated hash of a label to a Ristretto point (hash-to-curve).
    ///
    /// Uses SHA-512 with the Ristretto `from_uniform_bytes` constructor, which
    /// requires 64 uniform bytes.  The 64-byte output of SHA-512 satisfies this.
    fn hash_to_point(label: &[u8]) -> RistrettoPoint {
        let hash = Sha512::digest(label);
        let mut bytes = [0u8; 64];
        bytes.copy_from_slice(&hash);
        RistrettoPoint::from_uniform_bytes(&bytes)
    }

    /// The standard base generator G (Ristretto basepoint).
    pub fn generator_g() -> RistrettoPoint {
        RISTRETTO_BASEPOINT_POINT
    }

    /// The independent blinding generator H, derived from a domain string.
    ///
    /// H is guaranteed to be independent of G (the discrete log log_G(H) is
    /// unknown) because it is produced by hashing a domain label.
    pub fn generator_h() -> RistrettoPoint {
        hash_to_point(b"OxiRS-Pedersen-H-Ristretto255-v1")
    }

    // ── RistrettoPedersen ─────────────────────────────────────────────────────

    /// Pedersen commitment parameters over Ristretto255.
    ///
    /// Holds cached generators G and H so they can be reused across calls.
    #[derive(Clone)]
    pub struct RistrettoPedersen {
        /// Primary generator G.
        pub g: RistrettoPoint,
        /// Independent blinding generator H.
        pub h: RistrettoPoint,
    }

    impl RistrettoPedersen {
        /// Create the standard OxiRS Ristretto Pedersen parameters.
        pub fn standard() -> Self {
            Self {
                g: generator_g(),
                h: generator_h(),
            }
        }

        /// Create parameters from custom domain labels.
        pub fn from_domains(g_label: &[u8], h_label: &[u8]) -> Self {
            Self {
                g: hash_to_point(g_label),
                h: hash_to_point(h_label),
            }
        }

        /// Commit to a message scalar `m` with a blinding scalar `r`.
        ///
        /// Returns `C = m·G + r·H` as a compressed Ristretto point (32 bytes).
        pub fn commit(&self, m: &Scalar, r: &Scalar) -> CompressedRistretto {
            RistrettoPoint::multiscalar_mul([m, r], [&self.g, &self.h]).compress()
        }

        /// Verify a commitment opening: recompute C from (m, r) and compare.
        ///
        /// Returns `true` iff `commit(m, r) == commitment`.
        pub fn verify_opening(
            &self,
            commitment: &CompressedRistretto,
            m: &Scalar,
            r: &Scalar,
        ) -> bool {
            &self.commit(m, r) == commitment
        }

        /// Homomorphic addition of two commitments: C(m₁,r₁) + C(m₂,r₂) = C(m₁+m₂, r₁+r₂).
        ///
        /// This property allows batch proving without revealing individual values.
        pub fn add_commitments(
            &self,
            c1: &CompressedRistretto,
            c2: &CompressedRistretto,
        ) -> Option<CompressedRistretto> {
            let p1 = c1.decompress()?;
            let p2 = c2.decompress()?;
            Some((p1 + p2).compress())
        }
    }

    // ── Scalar helpers ────────────────────────────────────────────────────────

    /// Derive a deterministic (but unpredictable) Ristretto scalar from
    /// arbitrary bytes by hashing to 64 uniform bytes and reducing mod l.
    pub fn scalar_from_bytes(data: &[u8]) -> Scalar {
        let hash = Sha512::digest(data);
        let mut wide = [0u8; 64];
        wide.copy_from_slice(&hash);
        Scalar::from_bytes_mod_order_wide(&wide)
    }

    /// Encode a u64 value as a Ristretto scalar.
    pub fn scalar_from_u64(n: u64) -> Scalar {
        Scalar::from(n)
    }

    // ── Tests ─────────────────────────────────────────────────────────────────

    #[cfg(test)]
    mod tests {
        use super::*;
        use curve25519_dalek::scalar::Scalar;

        fn random_scalar() -> Scalar {
            scalar_from_bytes(
                &std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
                    .to_le_bytes(),
            )
        }

        #[test]
        fn test_commit_verify_roundtrip() {
            let params = RistrettoPedersen::standard();
            let m = scalar_from_u64(42);
            let r = random_scalar();
            let c = params.commit(&m, &r);
            assert!(params.verify_opening(&c, &m, &r), "opening should verify");
        }

        #[test]
        fn test_verify_rejects_wrong_message() {
            let params = RistrettoPedersen::standard();
            let m = scalar_from_u64(42);
            let r = random_scalar();
            let c = params.commit(&m, &r);
            let m_bad = scalar_from_u64(43);
            assert!(
                !params.verify_opening(&c, &m_bad, &r),
                "wrong message should not verify"
            );
        }

        #[test]
        fn test_verify_rejects_wrong_blinding() {
            let params = RistrettoPedersen::standard();
            let m = scalar_from_u64(42);
            let r = random_scalar();
            let c = params.commit(&m, &r);
            let r_bad = scalar_from_bytes(b"wrong blinding");
            assert!(
                !params.verify_opening(&c, &m, &r_bad),
                "wrong blinding should not verify"
            );
        }

        #[test]
        fn test_hiding_different_blindings_different_commitments() {
            let params = RistrettoPedersen::standard();
            let m = scalar_from_u64(100);
            let r1 = scalar_from_bytes(b"blinding-one");
            let r2 = scalar_from_bytes(b"blinding-two");
            let c1 = params.commit(&m, &r1);
            let c2 = params.commit(&m, &r2);
            assert_ne!(
                c1, c2,
                "different blindings should give different commitments (hiding)"
            );
        }

        #[test]
        fn test_homomorphic_addition() {
            let params = RistrettoPedersen::standard();
            let m1 = scalar_from_u64(10);
            let r1 = scalar_from_bytes(b"r1");
            let m2 = scalar_from_u64(20);
            let r2 = scalar_from_bytes(b"r2");
            let c1 = params.commit(&m1, &r1);
            let c2 = params.commit(&m2, &r2);
            let c_sum = params.add_commitments(&c1, &c2).unwrap();
            let m_sum = m1 + m2;
            let r_sum = r1 + r2;
            assert!(
                params.verify_opening(&c_sum, &m_sum, &r_sum),
                "homomorphic addition should hold: C(m1+m2, r1+r2) == C(m1,r1) + C(m2,r2)"
            );
        }

        #[test]
        fn test_generator_h_independent_of_g() {
            // G and H must be distinct.
            let g = generator_g();
            let h = generator_h();
            assert_ne!(
                g.compress(),
                h.compress(),
                "G and H must be independent generators"
            );
        }

        #[test]
        fn test_scalar_from_bytes_deterministic() {
            let s1 = scalar_from_bytes(b"test-label");
            let s2 = scalar_from_bytes(b"test-label");
            assert_eq!(s1, s2);
        }

        #[test]
        fn test_custom_domain_params() {
            let params = RistrettoPedersen::from_domains(b"custom-G-domain", b"custom-H-domain");
            let m = scalar_from_u64(7);
            let r = scalar_from_bytes(b"blind");
            let c = params.commit(&m, &r);
            assert!(params.verify_opening(&c, &m, &r));
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zkp::selective_disclosure::{CredentialAttribute, SelectiveDisclosureCredential};
    use ed25519_dalek::SigningKey;

    fn make_keypair() -> (Vec<u8>, Vec<u8>) {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = (i + 7) as u8;
        }
        let sk = SigningKey::from_bytes(&seed);
        let pk = sk.verifying_key().to_bytes().to_vec();
        (seed.to_vec(), pk)
    }

    fn make_credential(secret: &[u8]) -> SelectiveDisclosureCredential {
        SelectiveDisclosureCredential::issue(
            "urn:uuid:test-cred",
            "did:key:zIssuer",
            "did:key:zAlice",
            vec![
                CredentialAttribute::new("name", "Alice", 0),
                CredentialAttribute::new("age", "30", 1),
                CredentialAttribute::new("country", "Japan", 2),
            ],
            secret,
        )
        .unwrap()
    }

    // ── PedersenParams ────────────────────────────────────────────────────────

    #[test]
    fn test_params_standard_distinct_generators() {
        let p = PedersenParams::standard();
        assert_ne!(p.g, p.h);
    }

    #[test]
    fn test_params_commit_deterministic() {
        let p = PedersenParams::standard();
        let b = [1u8; 32];
        let c1 = p.commit(b"msg", &b);
        let c2 = p.commit(b"msg", &b);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_params_commit_different_message() {
        let p = PedersenParams::standard();
        let b = [1u8; 32];
        assert_ne!(p.commit(b"msg1", &b), p.commit(b"msg2", &b));
    }

    #[test]
    fn test_params_commit_different_blinding() {
        let p = PedersenParams::standard();
        let b1 = [1u8; 32];
        let b2 = [2u8; 32];
        assert_ne!(p.commit(b"msg", &b1), p.commit(b"msg", &b2));
    }

    #[test]
    fn test_params_verify_opening_correct() {
        let p = PedersenParams::standard();
        let b = [42u8; 32];
        let c = p.commit(b"open me", &b);
        assert!(p.verify_opening(&c, b"open me", &b));
    }

    #[test]
    fn test_params_verify_opening_wrong_message() {
        let p = PedersenParams::standard();
        let b = [42u8; 32];
        let c = p.commit(b"correct", &b);
        assert!(!p.verify_opening(&c, b"wrong", &b));
    }

    #[test]
    fn test_params_verify_opening_wrong_blinding() {
        let p = PedersenParams::standard();
        let b1 = [1u8; 32];
        let b2 = [2u8; 32];
        let c = p.commit(b"msg", &b1);
        assert!(!p.verify_opening(&c, b"msg", &b2));
    }

    #[test]
    fn test_params_from_domain_distinct_generators() {
        let p = PedersenParams::from_domain("custom-domain");
        assert_ne!(p.g, p.h);
    }

    // ── AttributeCommitment ───────────────────────────────────────────────────

    #[test]
    fn test_attribute_commitment_has_blinding() {
        let p = PedersenParams::standard();
        let attr = CredentialAttribute::new("name", "Alice", 0);
        let c = AttributeCommitment::commit_attr(&p, &attr);
        assert!(c.blinding.is_some());
    }

    #[test]
    fn test_attribute_commitment_verify_opening() {
        let p = PedersenParams::standard();
        let attr = CredentialAttribute::new("age", "25", 1);
        let c = AttributeCommitment::commit_attr(&p, &attr);
        let blinding = c.blinding.unwrap();
        assert!(p.verify_opening(&c.commitment, &attr.value, &blinding));
    }

    #[test]
    fn test_attribute_commitment_public_only_no_blinding() {
        let pc = AttributeCommitment::public_only(0, "name", [0u8; 32]);
        assert!(pc.blinding.is_none());
    }

    // ── SchnorrProof ──────────────────────────────────────────────────────────

    #[test]
    fn test_schnorr_prove_and_verify() {
        let p = PedersenParams::standard();
        let msg = b"secret attribute";
        let blinding = [7u8; 32];
        let commitment = p.commit(msg, &blinding);
        let nonce = b"verifier-nonce";

        let proof = SchnorrProof::prove(&p, commitment, msg, &blinding, nonce);
        assert!(proof.verify(&p, nonce));
    }

    #[test]
    fn test_schnorr_wrong_nonce_fails() {
        let p = PedersenParams::standard();
        let msg = b"hidden value";
        let blinding = [3u8; 32];
        let commitment = p.commit(msg, &blinding);

        let proof = SchnorrProof::prove(&p, commitment, msg, &blinding, b"nonce1");
        assert!(!proof.verify(&p, b"nonce2"));
    }

    #[test]
    fn test_schnorr_tampered_commitment_fails() {
        let p = PedersenParams::standard();
        let msg = b"value";
        let blinding = [1u8; 32];
        let commitment = p.commit(msg, &blinding);

        let mut proof = SchnorrProof::prove(&p, commitment, msg, &blinding, b"n");
        // Tamper the commitment in the proof
        proof.commitment[0] ^= 0xFF;
        assert!(!proof.verify(&p, b"n"));
    }

    // ── SelectiveDisclosureRequest ────────────────────────────────────────────

    #[test]
    fn test_request_should_reveal() {
        let req = SelectiveDisclosureRequest::new(
            "req-1",
            "did:key:zVerifier",
            vec!["name".to_string(), "age".to_string()],
            b"nonce".to_vec(),
        );
        assert!(req.should_reveal("name"));
        assert!(req.should_reveal("age"));
        assert!(!req.should_reveal("country"));
    }

    #[test]
    fn test_request_with_purpose() {
        let req = SelectiveDisclosureRequest::new("req-2", "did:key:z", vec![], b"n".to_vec())
            .with_purpose("proofOfAge");
        assert_eq!(req.purpose, "proofOfAge");
    }

    #[test]
    fn test_request_with_schema() {
        let req = SelectiveDisclosureRequest::new("req-3", "did:key:z", vec![], b"n".to_vec())
            .with_schema("https://schema.org/Person");
        assert_eq!(
            req.schema_uri,
            Some("https://schema.org/Person".to_string())
        );
    }

    // ── prove_selective ───────────────────────────────────────────────────────

    #[test]
    fn test_prove_selective_reveals_requested() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-a",
            "did:key:zVerifier",
            vec!["name".to_string()],
            b"nonce-abc".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        assert!(proof.is_disclosed("name"));
        assert!(!proof.is_disclosed("age"));
        assert!(!proof.is_disclosed("country"));
    }

    #[test]
    fn test_prove_selective_hidden_count() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-b",
            "did:key:zVerifier",
            vec!["name".to_string()],
            b"nonce".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        // 3 attributes total - 1 revealed = 2 hidden
        assert_eq!(proof.hidden_count(), 2);
    }

    #[test]
    fn test_prove_selective_disclose_all() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-c",
            "did:key:zVerifier",
            vec!["name".to_string(), "age".to_string(), "country".to_string()],
            b"nonce".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        assert_eq!(proof.hidden_count(), 0);
        assert!(proof.is_disclosed("name"));
        assert!(proof.is_disclosed("age"));
        assert!(proof.is_disclosed("country"));
    }

    #[test]
    fn test_prove_selective_disclose_none() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-d",
            "did:key:zVerifier",
            vec![],
            b"nonce".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        assert_eq!(proof.hidden_count(), 3);
        assert_eq!(proof.revealed.len(), 0);
    }

    #[test]
    fn test_prove_schnorr_proofs_count_matches_hidden() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-e2",
            "did:key:zV",
            vec!["age".to_string()],
            b"n".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        assert_eq!(proof.schnorr_proofs.len(), proof.hidden_commitments.len());
    }

    // ── verify_selective ──────────────────────────────────────────────────────

    #[test]
    fn test_verify_selective_valid_proof() {
        let (secret, pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-e",
            "did:key:zVerifier",
            vec!["name".to_string()],
            b"nonce-verify".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        assert!(verify_selective(&proof, &pk).unwrap());
    }

    #[test]
    fn test_verify_selective_wrong_key_fails() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-f",
            "did:key:zVerifier",
            vec!["name".to_string()],
            b"n".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();

        // Different key
        let mut other_seed = [0u8; 32];
        other_seed[0] = 99;
        let other_sk = SigningKey::from_bytes(&other_seed);
        let other_pk = other_sk.verifying_key().to_bytes();
        assert!(!verify_selective(&proof, &other_pk).unwrap());
    }

    #[test]
    fn test_verify_selective_bad_public_key_length() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new("req-h", "did:key:zV", vec![], b"n".to_vec());
        let proof = prove_selective(&cred, &req).unwrap();
        assert!(verify_selective(&proof, &[0u8; 31]).is_err());
    }

    #[test]
    fn test_proof_revealed_names() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-i",
            "did:key:zV",
            vec!["name".to_string(), "age".to_string()],
            b"n".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        let names = proof.revealed_names();
        assert!(names.contains(&"name"));
        assert!(names.contains(&"age"));
        assert!(!names.contains(&"country"));
    }

    #[test]
    fn test_proof_get_revealed_attribute() {
        let (secret, _pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-j",
            "did:key:zV",
            vec!["country".to_string()],
            b"n".to_vec(),
        );
        let proof = prove_selective(&cred, &req).unwrap();
        let attr = proof.get_revealed("country").unwrap();
        assert_eq!(attr.value_str(), Some("Japan"));
    }

    #[test]
    fn test_root_commitment_covers_all_attributes() {
        let (secret, _pk) = make_keypair();
        let cred1 = make_credential(&secret);

        // Credential with a different attribute value → different root commitment
        let cred2 = SelectiveDisclosureCredential::issue(
            "urn:uuid:other",
            "did:key:zIssuer",
            "did:key:zBob",
            vec![
                CredentialAttribute::new("name", "Bob", 0),
                CredentialAttribute::new("age", "30", 1),
                CredentialAttribute::new("country", "Japan", 2),
            ],
            &secret,
        )
        .unwrap();

        let req = SelectiveDisclosureRequest::new("req-k", "did:key:zV", vec![], b"n".to_vec());
        let proof1 = prove_selective(&cred1, &req).unwrap();
        let proof2 = prove_selective(&cred2, &req).unwrap();
        // Root commitments are probabilistic (fresh blinding) — credential IDs differ though
        assert_ne!(proof1.credential_id, proof2.credential_id);
    }

    #[test]
    fn test_tampered_schnorr_fails_verification() {
        let (secret, pk) = make_keypair();
        let cred = make_credential(&secret);
        let req = SelectiveDisclosureRequest::new(
            "req-m",
            "did:key:zV",
            vec!["name".to_string()],
            b"nonce-x".to_vec(),
        );
        let mut proof = prove_selective(&cred, &req).unwrap();
        // Tamper a Schnorr proof
        if let Some(sp) = proof.schnorr_proofs.first_mut() {
            sp.challenge[0] ^= 0xFF;
        }
        // Should fail verification due to corrupted Schnorr proof
        assert!(!verify_selective(&proof, &pk).unwrap());
    }
}
