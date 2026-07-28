//! Pure-Rust X.509 certificate signature verification.
//!
//! This module implements the cryptographic half of client-certificate / mTLS
//! trust checking: given a leaf certificate and a candidate CA certificate, it
//! answers *"was this leaf actually signed by that CA's private key?"* by
//! verifying the leaf's signature over its `TBSCertificate` against the CA's
//! public key.
//!
//! # Why this exists
//!
//! The previous implementation compared the leaf's Issuer DN string against the
//! CA's Subject DN string and treated equality as proof of signing. X.509
//! Issuer/Subject DNs are **public, non-secret** strings that appear verbatim in
//! every certificate a CA issues, so an attacker could mint a fully self-signed
//! certificate whose Issuer field copies a trusted CA's Subject DN and have it
//! accepted — a complete mTLS bypass. DN equality is necessary (it selects which
//! CA to check against) but **never sufficient**; only a public-key signature
//! check proves issuance.
//!
//! # Why not `x509-parser`'s `verify` feature
//!
//! `x509-parser`'s built-in `verify_signature` is gated behind its `verify`
//! feature, which pulls in `ring` (C/assembly). The COOLJAPAN Pure Rust policy
//! forbids that, so signature verification is implemented here on top of the
//! pure-Rust RustCrypto crates (`rsa`, `p256`, `ed25519-dalek`).
//!
//! # Supported signature algorithms
//!
//! - RSASSA-PKCS1-v1_5 with SHA-256 / SHA-384 / SHA-512
//! - ECDSA (NIST P-256) with SHA-256
//! - Ed25519
//!
//! Any other algorithm yields `Ok(false)` (fail-closed): the certificate is not
//! considered signed by this CA, so trust falls through to the other checks in
//! [`super::certificate`] and, failing those, authentication is rejected.

use crate::error::FusekiResult;
use tracing::debug;
use x509_parser::oid_registry::{
    OID_EC_P256, OID_PKCS1_SHA256WITHRSA, OID_PKCS1_SHA384WITHRSA, OID_PKCS1_SHA512WITHRSA,
    OID_SIG_ECDSA_WITH_SHA256, OID_SIG_ED25519,
};
use x509_parser::prelude::*;

/// Verify that `cert` was cryptographically signed by `ca_cert`.
///
/// Returns `Ok(true)` only when:
/// 1. the leaf's Issuer DN equals the CA's Subject DN (name chaining), and
/// 2. the leaf's signature over its `TBSCertificate` verifies against the CA's
///    public key using the leaf's declared signature algorithm.
///
/// Any name mismatch, unsupported algorithm, malformed key, or failed signature
/// check returns `Ok(false)` (fail-closed). `Err` is reserved for genuine
/// internal errors (there are currently none — all failure modes are `Ok`).
pub fn verify_certificate_signed_by(
    cert: &X509Certificate<'_>,
    ca_cert: &X509Certificate<'_>,
) -> FusekiResult<bool> {
    // Name chaining: the leaf must claim to be issued by this CA. This is a fast
    // pre-filter so we only attempt the (relatively expensive) crypto against the
    // matching CA — it is NOT, on its own, evidence of issuance.
    if cert.issuer().to_string() != ca_cert.subject().to_string() {
        return Ok(false);
    }

    let tbs = cert.tbs_certificate.as_ref();
    let signature = cert.signature_value.data.as_ref();
    let sig_alg = &cert.signature_algorithm.algorithm;
    let spki = ca_cert.public_key();

    let verified = if *sig_alg == OID_PKCS1_SHA256WITHRSA {
        verify_rsa_sha256(spki, tbs, signature)
    } else if *sig_alg == OID_PKCS1_SHA384WITHRSA {
        verify_rsa_sha384(spki, tbs, signature)
    } else if *sig_alg == OID_PKCS1_SHA512WITHRSA {
        verify_rsa_sha512(spki, tbs, signature)
    } else if *sig_alg == OID_SIG_ECDSA_WITH_SHA256 {
        verify_ecdsa_p256(spki, tbs, signature)
    } else if *sig_alg == OID_SIG_ED25519 {
        verify_ed25519(spki, tbs, signature)
    } else {
        debug!(
            algorithm = ?sig_alg,
            "Unsupported certificate signature algorithm; treating as untrusted"
        );
        false
    };

    Ok(verified)
}

/// Parse the CA's RSA public key from its SPKI, or `None` if it is not a valid
/// PKCS#1 RSA key.
fn rsa_public_key_from_spki(spki: &SubjectPublicKeyInfo<'_>) -> Option<rsa::RsaPublicKey> {
    use rsa::pkcs1::DecodeRsaPublicKey;
    // The SPKI subjectPublicKey BIT STRING for RSA is the DER encoding of a
    // PKCS#1 `RSAPublicKey`.
    match rsa::RsaPublicKey::from_pkcs1_der(spki.subject_public_key.data.as_ref()) {
        Ok(key) => Some(key),
        Err(e) => {
            debug!(error = %e, "Failed to parse RSA public key from CA SPKI");
            None
        }
    }
}

/// Verify an RSASSA-PKCS1-v1_5 signature over `tbs` using SHA-256.
fn verify_rsa_sha256(spki: &SubjectPublicKeyInfo<'_>, tbs: &[u8], signature: &[u8]) -> bool {
    use rsa::sha2::{Digest, Sha256};
    let Some(public_key) = rsa_public_key_from_spki(spki) else {
        return false;
    };
    let hashed = Sha256::digest(tbs);
    public_key
        .verify(rsa::Pkcs1v15Sign::new::<Sha256>(), &hashed, signature)
        .is_ok()
}

/// Verify an RSASSA-PKCS1-v1_5 signature over `tbs` using SHA-384.
fn verify_rsa_sha384(spki: &SubjectPublicKeyInfo<'_>, tbs: &[u8], signature: &[u8]) -> bool {
    use rsa::sha2::{Digest, Sha384};
    let Some(public_key) = rsa_public_key_from_spki(spki) else {
        return false;
    };
    let hashed = Sha384::digest(tbs);
    public_key
        .verify(rsa::Pkcs1v15Sign::new::<Sha384>(), &hashed, signature)
        .is_ok()
}

/// Verify an RSASSA-PKCS1-v1_5 signature over `tbs` using SHA-512.
fn verify_rsa_sha512(spki: &SubjectPublicKeyInfo<'_>, tbs: &[u8], signature: &[u8]) -> bool {
    use rsa::sha2::{Digest, Sha512};
    let Some(public_key) = rsa_public_key_from_spki(spki) else {
        return false;
    };
    let hashed = Sha512::digest(tbs);
    public_key
        .verify(rsa::Pkcs1v15Sign::new::<Sha512>(), &hashed, signature)
        .is_ok()
}

/// Verify an ECDSA/P-256 signature (SHA-256, ASN.1 DER-encoded signature).
fn verify_ecdsa_p256(spki: &SubjectPublicKeyInfo<'_>, tbs: &[u8], signature: &[u8]) -> bool {
    use p256::ecdsa::signature::Verifier;

    // Confirm the CA key is actually on curve P-256; anything else cannot be
    // verified by this path.
    let curve_ok = spki
        .algorithm
        .parameters
        .as_ref()
        .and_then(|p| p.as_oid().ok())
        .map(|oid| oid == OID_EC_P256)
        .unwrap_or(false);
    if !curve_ok {
        debug!("ECDSA-with-SHA256 certificate but CA key is not P-256; untrusted");
        return false;
    }

    // The SPKI subjectPublicKey for EC is the SEC1-encoded point.
    let verifying_key =
        match p256::ecdsa::VerifyingKey::from_sec1_bytes(spki.subject_public_key.data.as_ref()) {
            Ok(key) => key,
            Err(e) => {
                debug!(error = %e, "Failed to parse P-256 public key from CA SPKI");
                return false;
            }
        };

    let sig = match p256::ecdsa::DerSignature::try_from(signature) {
        Ok(sig) => sig,
        Err(e) => {
            debug!(error = %e, "Failed to parse ECDSA signature");
            return false;
        }
    };

    verifying_key.verify(tbs, &sig).is_ok()
}

/// Verify an Ed25519 signature.
fn verify_ed25519(spki: &SubjectPublicKeyInfo<'_>, tbs: &[u8], signature: &[u8]) -> bool {
    use ed25519_dalek::Verifier;

    let key_bytes: [u8; 32] = match spki.subject_public_key.data.as_ref().try_into() {
        Ok(bytes) => bytes,
        Err(_) => {
            debug!("Ed25519 CA public key is not 32 bytes; untrusted");
            return false;
        }
    };
    let verifying_key = match ed25519_dalek::VerifyingKey::from_bytes(&key_bytes) {
        Ok(key) => key,
        Err(e) => {
            debug!(error = %e, "Failed to parse Ed25519 public key from CA SPKI");
            return false;
        }
    };

    let sig_bytes: [u8; 64] = match signature.try_into() {
        Ok(bytes) => bytes,
        Err(_) => {
            debug!("Ed25519 signature is not 64 bytes; untrusted");
            return false;
        }
    };
    let sig = ed25519_dalek::Signature::from_bytes(&sig_bytes);

    verifying_key.verify(tbs, &sig).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, Issuer, KeyPair, PKCS_ED25519};
    use x509_parser::prelude::FromDer;

    /// Build a CA (self-signed) and a leaf certificate signed by it, plus the
    /// raw DER of both. `alg_ed25519` selects Ed25519 vs the default ECDSA
    /// P-256 for both keys.
    fn make_ca_and_leaf(ed25519: bool) -> (Vec<u8>, Vec<u8>) {
        let (ca_key, leaf_key) = if ed25519 {
            (
                KeyPair::generate_for(&PKCS_ED25519).expect("ca key"),
                KeyPair::generate_for(&PKCS_ED25519).expect("leaf key"),
            )
        } else {
            (
                KeyPair::generate().expect("ca key"),
                KeyPair::generate().expect("leaf key"),
            )
        };

        let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("ca params");
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "OxiRS Test Root CA");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).expect("ca self-signed");
        let ca_der = ca_cert.der().to_vec();

        let issuer = Issuer::new(ca_params, ca_key);

        let mut leaf_params = CertificateParams::new(Vec::<String>::new()).expect("leaf params");
        leaf_params
            .distinguished_name
            .push(DnType::CommonName, "alice");
        let leaf_cert = leaf_params
            .signed_by(&leaf_key, &issuer)
            .expect("leaf signed by ca");
        let leaf_der = leaf_cert.der().to_vec();

        (ca_der, leaf_der)
    }

    #[test]
    fn regression_genuine_ecdsa_chain_verifies() {
        let (ca_der, leaf_der) = make_ca_and_leaf(false);
        let (_, ca) = X509Certificate::from_der(&ca_der).expect("parse ca");
        let (_, leaf) = X509Certificate::from_der(&leaf_der).expect("parse leaf");

        assert!(
            verify_certificate_signed_by(&leaf, &ca).expect("verify"),
            "a leaf genuinely signed by the CA must verify"
        );
    }

    #[test]
    fn regression_genuine_ed25519_chain_verifies() {
        let (ca_der, leaf_der) = make_ca_and_leaf(true);
        let (_, ca) = X509Certificate::from_der(&ca_der).expect("parse ca");
        let (_, leaf) = X509Certificate::from_der(&leaf_der).expect("parse leaf");

        assert!(
            verify_certificate_signed_by(&leaf, &ca).expect("verify"),
            "an Ed25519 leaf genuinely signed by the CA must verify"
        );
    }

    /// The core attack the finding describes: a forged certificate whose Issuer
    /// DN copies a trusted CA's Subject DN verbatim, but which is signed by the
    /// attacker's own key. The old DN-string comparison accepted this; real
    /// signature verification must reject it.
    #[test]
    fn regression_forged_issuer_dn_is_rejected() {
        // Genuine CA.
        let ca_key = KeyPair::generate().expect("ca key");
        let mut ca_params = CertificateParams::new(Vec::<String>::new()).expect("ca params");
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "OxiRS Test Root CA");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let ca_cert = ca_params.self_signed(&ca_key).expect("ca self-signed");
        let ca_der = ca_cert.der().to_vec();

        // Attacker builds an "issuer" whose DN copies the trusted CA's subject
        // DN, but backed by the attacker's own key.
        let attacker_key = KeyPair::generate().expect("attacker key");
        let mut forged_issuer_params =
            CertificateParams::new(Vec::<String>::new()).expect("forged issuer params");
        forged_issuer_params
            .distinguished_name
            .push(DnType::CommonName, "OxiRS Test Root CA");
        forged_issuer_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let forged_issuer = Issuer::new(forged_issuer_params, attacker_key);

        // Leaf whose Issuer field now reads "CN=OxiRS Test Root CA" but is
        // signed by the attacker, not the real CA.
        let leaf_key = KeyPair::generate().expect("leaf key");
        let mut leaf_params = CertificateParams::new(Vec::<String>::new()).expect("leaf params");
        leaf_params
            .distinguished_name
            .push(DnType::CommonName, "alice");
        let forged_leaf = leaf_params
            .signed_by(&leaf_key, &forged_issuer)
            .expect("forged leaf");
        let forged_der = forged_leaf.der().to_vec();

        let (_, ca) = X509Certificate::from_der(&ca_der).expect("parse ca");
        let (_, forged) = X509Certificate::from_der(&forged_der).expect("parse forged");

        // The Issuer DN matches the CA Subject DN (the old check would pass)...
        assert_eq!(
            forged.issuer().to_string(),
            ca.subject().to_string(),
            "test precondition: forged issuer DN copies the CA subject DN"
        );
        // ...but the signature was NOT produced by the CA key, so it must fail.
        assert!(
            !verify_certificate_signed_by(&forged, &ca).expect("verify"),
            "a forged cert with a copied Issuer DN must NOT verify against the real CA"
        );
    }

    #[test]
    fn regression_wrong_ca_is_rejected() {
        // Two independent CAs; a leaf signed by CA #1 must not verify against
        // CA #2 (different subject DN and different key).
        let (_ca1_der, leaf_der) = make_ca_and_leaf(false);

        let other_key = KeyPair::generate().expect("other ca key");
        let mut other_params =
            CertificateParams::new(Vec::<String>::new()).expect("other ca params");
        other_params
            .distinguished_name
            .push(DnType::CommonName, "Unrelated CA");
        other_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        let other_ca = other_params.self_signed(&other_key).expect("other ca");
        let other_der = other_ca.der().to_vec();

        let (_, other) = X509Certificate::from_der(&other_der).expect("parse other ca");
        let (_, leaf) = X509Certificate::from_der(&leaf_der).expect("parse leaf");

        assert!(
            !verify_certificate_signed_by(&leaf, &other).expect("verify"),
            "a leaf signed by a different CA must not verify against an unrelated CA"
        );
    }
}
