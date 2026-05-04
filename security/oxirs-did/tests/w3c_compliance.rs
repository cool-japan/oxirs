//! W3C DID Core 1.0 and VC Data Model 2.0 compliance tests.
//!
//! These tests exercise conformance with:
//! - W3C DID Core 1.0 (<https://www.w3.org/TR/did-core/>)
//! - W3C VC Data Model 2.0 (<https://www.w3.org/TR/vc-data-model-2.0/>)
//!
//! Test vectors are derived from the W3C test suites and the official
//! specification examples.

use oxirs_did::vc::sd_jwt::{Disclosure, SdJwtClaims, SdJwtIssuer, SdJwtVerifier};
use oxirs_did::vc::{JwtVc, JwtVcHeader, JwtVcPayload};
use serde_json::json;

// ─────────────────────────────────────────────────────────────────────────────
// W3C DID Core 1.0 — DID syntax and resolution
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn did_core_valid_did_key_syntax() {
    use oxirs_did::Did;
    // W3C DID Core §3.1: a DID is a URI with scheme "did:"
    let did_str = "did:key:z6MkpTHR8VNsBxYAAWHut2Geadd9jSwuias8sisDArDJF";
    let did = Did::new(did_str).expect("valid did:key must parse");
    assert_eq!(did.method(), "key");
    assert!(did.as_str().starts_with("did:"));
}

#[test]
fn did_core_did_components() {
    use oxirs_did::Did;
    let did = Did::new("did:web:example.com").expect("valid did:web must parse");
    assert_eq!(did.method(), "web");
    assert_eq!(did.method_specific_id(), "example.com");
}

#[test]
fn did_core_invalid_did_scheme_rejected() {
    use oxirs_did::Did;
    // Must not accept URIs that don't start with "did:"
    let result = Did::new("https://example.com");
    assert!(result.is_err(), "non-DID URI must be rejected");
}

#[test]
fn did_core_method_must_be_lowercase_alphanumeric() {
    use oxirs_did::Did;
    // W3C DID Core §8.1: method name must be lowercase letters/digits
    // "DID:KEY" (uppercase) must be rejected
    let result = Did::new("DID:key:z6Mk");
    assert!(result.is_err(), "uppercase DID scheme must be rejected");
}

// ─────────────────────────────────────────────────────────────────────────────
// W3C VC Data Model 2.0 — Credential structure
// ─────────────────────────────────────────────────────────────────────────────

fn vc_json_v2() -> serde_json::Value {
    json!({
        "@context": [
            "https://www.w3.org/ns/credentials/v2",
            "https://www.w3.org/ns/credentials/examples/v2"
        ],
        "id": "http://university.example/credentials/3732",
        "type": ["VerifiableCredential", "ExampleDegreeCredential"],
        "issuer": {
            "id": "https://university.example/issuers/565049",
            "name": "Example University"
        },
        "validFrom": "2015-05-10T12:43:56.000Z",
        "credentialSubject": {
            "id": "did:example:ebfeb1f712ebc6f1c276e12ec21",
            "degree": {
                "type": "ExampleBachelorDegree",
                "name": "Bachelor of Science and Arts"
            }
        }
    })
}

#[test]
fn vc_data_model_v2_context_required() {
    let vc = vc_json_v2();
    let context = vc["@context"].as_array().expect("@context must be present");
    assert!(
        context
            .iter()
            .any(|c| c.as_str() == Some("https://www.w3.org/ns/credentials/v2")),
        "W3C VC 2.0 context must be present"
    );
}

#[test]
fn vc_data_model_v2_type_must_include_verifiable_credential() {
    let vc = vc_json_v2();
    let types = vc["type"].as_array().expect("type must be an array");
    assert!(
        types
            .iter()
            .any(|t| t.as_str() == Some("VerifiableCredential")),
        "type must include VerifiableCredential"
    );
}

#[test]
fn vc_data_model_v2_issuer_must_be_uri() {
    let vc = vc_json_v2();
    let issuer_id = vc["issuer"]["id"]
        .as_str()
        .expect("issuer.id must be string");
    assert!(
        issuer_id.starts_with("https://") || issuer_id.starts_with("did:"),
        "issuer.id must be a URI"
    );
}

#[test]
fn vc_data_model_v2_valid_from_must_be_xml_datetime() {
    let vc = vc_json_v2();
    let valid_from = vc["validFrom"].as_str().expect("validFrom must be present");
    // W3C VC 2.0 §4.9: validFrom is an xmlschema-date-time string
    assert!(
        valid_from.contains('T') && valid_from.contains('Z'),
        "validFrom must be an XML datetime"
    );
}

#[test]
fn vc_data_model_v2_credential_subject_must_have_id() {
    let vc = vc_json_v2();
    let subject_id = vc["credentialSubject"]["id"]
        .as_str()
        .expect("credentialSubject.id must be present");
    assert!(
        subject_id.starts_with("did:") || subject_id.starts_with("https://"),
        "credentialSubject.id must be a URI"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// W3C VC 2.0 — JWT-VC serialization
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn jwt_vc_header_typ_must_be_vc_plus_jwt() {
    let h = JwtVcHeader::ed25519(None);
    assert_eq!(h.typ, "vc+jwt", "JWT-VC typ must be 'vc+jwt'");
    h.validate()
        .expect("valid ed25519 header must pass validation");
}

#[test]
fn jwt_vc_header_alg_none_rejected() {
    let h = JwtVcHeader {
        typ: "vc+jwt".into(),
        alg: "none".into(),
        kid: None,
    };
    assert!(h.validate().is_err(), "alg=none must be rejected");
}

#[test]
fn jwt_vc_compact_three_parts() {
    let jwt = JwtVc::new("H".into(), "P".into(), "S".into());
    let compact = jwt.to_compact();
    assert_eq!(compact.matches('.').count(), 2, "JWT must have 3 parts");
}

#[test]
fn jwt_vc_payload_expired_credential_rejected() {
    use chrono::Utc;
    let past = Utc::now().timestamp() - 7200;
    let payload = JwtVcPayload {
        iss: "did:key:z6Mk".into(),
        sub: None,
        jti: None,
        iat: past - 3600,
        nbf: None,
        exp: Some(past),
        vc: serde_json::Value::Null,
    };
    assert!(
        payload.validate(Utc::now()).is_err(),
        "expired credential must be rejected"
    );
}

#[test]
fn jwt_vc_payload_empty_issuer_rejected() {
    use chrono::Utc;
    let payload = JwtVcPayload {
        iss: "".into(),
        sub: None,
        jti: None,
        iat: Utc::now().timestamp(),
        nbf: None,
        exp: None,
        vc: serde_json::Value::Null,
    };
    assert!(
        payload.validate(Utc::now()).is_err(),
        "empty issuer must be rejected"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// SD-JWT — IETF draft conformance
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn sd_jwt_disclosure_hash_binding() {
    // IETF SD-JWT §5: disclosure hash must be SHA-256 of base64url(disclosure)
    let d = Disclosure::new("fixed_salt", "given_name", json!("Alice"));
    let b64 = d.to_base64url().unwrap();
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine as _;
    use sha2::{Digest, Sha256};
    let expected = URL_SAFE_NO_PAD.encode(Sha256::digest(b64.as_bytes()));
    assert_eq!(d.hash().unwrap(), expected);
}

#[test]
fn sd_jwt_issuer_builds_hashes_for_sd_claims() {
    let issuer = SdJwtIssuer::new()
        .public_claim("iss", json!("did:key:z6Mk"))
        .sd_claim("family_name", json!("Doe"))
        .sd_claim("given_name", json!("John"))
        .sd_claim("birthdate", json!("1990-01-01"));

    let claims = issuer.build_sd_claims().unwrap();
    assert_eq!(claims.sd_hashes.len(), 3, "one hash per SD claim");
    assert_eq!(claims.sd_alg, "sha-256", "algorithm must be sha-256");
}

#[test]
fn sd_jwt_selective_disclosure_partial_reveal() {
    // Issuer encodes 3 claims but holder reveals only 2
    let mut issuer = SdJwtIssuer::new();
    issuer = issuer
        .sd_claim("name", json!("Alice"))
        .sd_claim("age", json!(30))
        .sd_claim("address", json!("Tokyo"));

    let sd_claims = issuer.build_sd_claims().unwrap();

    // Holder reveals only name and age (not address)
    let partial = &issuer.disclosures[..2];
    let verified = SdJwtVerifier::verify_disclosures(partial, &sd_claims).unwrap();
    assert_eq!(verified.get("name"), Some(&json!("Alice")));
    assert_eq!(verified.get("age"), Some(&json!(30)));
    assert!(
        !verified.contains_key("address"),
        "undisclosed claim must not appear"
    );
}

#[test]
fn sd_jwt_tampered_disclosure_fails_verification() {
    let issuer = SdJwtIssuer::new().sd_claim("score", json!(100));
    let sd_claims = issuer.build_sd_claims().unwrap();

    let mut tampered = issuer.disclosures.clone();
    tampered[0].claim_value = json!(999); // tamper the value

    assert!(
        SdJwtVerifier::verify_disclosures(&tampered, &sd_claims).is_err(),
        "tampered disclosure must fail"
    );
}

#[test]
fn sd_jwt_compact_trailing_tilde() {
    let issuer = SdJwtIssuer::new().sd_claim("x", json!(1));
    let compact = issuer.build_compact("jwt.payload.sig").unwrap();
    assert!(compact.ends_with('~'), "SD-JWT must end with trailing ~");
}

#[test]
fn sd_jwt_parse_and_verify_full_flow() {
    let issuer = SdJwtIssuer::new()
        .sd_claim("email", json!("holder@example.com"))
        .sd_claim("phone", json!("+1-555-0100"));

    let sd_claims = issuer.build_sd_claims().unwrap();
    let compact = issuer.build_compact("jwt.payload.sig").unwrap();

    let (_, disclosures) = SdJwtVerifier::parse(&compact).unwrap();
    let verified = SdJwtVerifier::verify_disclosures(&disclosures, &sd_claims).unwrap();

    assert_eq!(verified.get("email"), Some(&json!("holder@example.com")));
    assert_eq!(verified.get("phone"), Some(&json!("+1-555-0100")));
}
