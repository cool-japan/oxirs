use crate::proof_purpose_registry::{ProofPurposeRegistry, PurposeMetadata};
use crate::proof_purpose_types::{
    CustomPurposeValidator, DidRelationships, ProofEntry, ProofPurposeKind, PurposeError,
    ValidationContext,
};
use crate::proof_purpose_verifier::ProofPurposeValidator;

fn make_relationships() -> DidRelationships {
    let mut rels = DidRelationships::new("did:example:alice");
    rels.add_authentication("did:example:alice#key-1");
    rels.add_assertion_method("did:example:alice#key-2");
    rels.add_key_agreement("did:example:alice#key-3");
    rels.add_capability_invocation("did:example:alice#key-4");
    rels.add_capability_delegation("did:example:alice#key-5");
    rels
}

fn make_context() -> ValidationContext {
    ValidationContext {
        challenge: Some("challenge-123".to_string()),
        domain: Some("example.com".to_string()),
        current_time_ms: 1_700_000_000_000,
    }
}

fn make_entry(purpose: ProofPurposeKind, vm: &str) -> ProofEntry {
    ProofEntry {
        purpose,
        verification_method: vm.to_string(),
        created_ms: 1_699_999_000_000,
        expires_ms: Some(1_700_100_000_000),
        challenge: Some("challenge-123".to_string()),
        domain: Some("example.com".to_string()),
        nonce: None,
    }
}

#[test]
fn test_purpose_kind_as_str() {
    assert_eq!(ProofPurposeKind::Authentication.as_str(), "authentication");
    assert_eq!(
        ProofPurposeKind::AssertionMethod.as_str(),
        "assertionMethod"
    );
    assert_eq!(ProofPurposeKind::KeyAgreement.as_str(), "keyAgreement");
    assert_eq!(
        ProofPurposeKind::CapabilityInvocation.as_str(),
        "capabilityInvocation"
    );
    assert_eq!(
        ProofPurposeKind::CapabilityDelegation.as_str(),
        "capabilityDelegation"
    );
}

#[test]
fn test_purpose_kind_custom() {
    let custom = ProofPurposeKind::Custom("https://example.com/myPurpose".to_string());
    assert_eq!(custom.as_str(), "https://example.com/myPurpose");
    assert!(!custom.is_standard());
}

#[test]
fn test_purpose_kind_is_standard() {
    assert!(ProofPurposeKind::Authentication.is_standard());
    assert!(ProofPurposeKind::AssertionMethod.is_standard());
    assert!(ProofPurposeKind::KeyAgreement.is_standard());
    assert!(ProofPurposeKind::CapabilityInvocation.is_standard());
    assert!(ProofPurposeKind::CapabilityDelegation.is_standard());
}

#[test]
fn test_purpose_kind_from_str_lossy() {
    assert_eq!(
        ProofPurposeKind::from_str_lossy("authentication"),
        ProofPurposeKind::Authentication
    );
    assert_eq!(
        ProofPurposeKind::from_str_lossy("assertionMethod"),
        ProofPurposeKind::AssertionMethod
    );
    assert_eq!(
        ProofPurposeKind::from_str_lossy("keyAgreement"),
        ProofPurposeKind::KeyAgreement
    );
    assert_eq!(
        ProofPurposeKind::from_str_lossy("capabilityInvocation"),
        ProofPurposeKind::CapabilityInvocation
    );
    assert_eq!(
        ProofPurposeKind::from_str_lossy("capabilityDelegation"),
        ProofPurposeKind::CapabilityDelegation
    );
    let c = ProofPurposeKind::from_str_lossy("custom:xyz");
    assert_eq!(c, ProofPurposeKind::Custom("custom:xyz".to_string()));
}

#[test]
fn test_purpose_kind_display() {
    assert_eq!(
        format!("{}", ProofPurposeKind::Authentication),
        "authentication"
    );
    assert_eq!(
        format!("{}", ProofPurposeKind::Custom("urn:x".to_string())),
        "urn:x"
    );
}

#[test]
fn test_did_relationships_new() {
    let rels = DidRelationships::new("did:example:bob");
    assert_eq!(rels.controller, "did:example:bob");
    assert_eq!(rels.total_entries(), 0);
}

#[test]
fn test_did_relationships_add_and_check() {
    let rels = make_relationships();
    assert!(rels.is_authorised(&ProofPurposeKind::Authentication, "did:example:alice#key-1"));
    assert!(rels.is_authorised(
        &ProofPurposeKind::AssertionMethod,
        "did:example:alice#key-2"
    ));
    assert!(rels.is_authorised(&ProofPurposeKind::KeyAgreement, "did:example:alice#key-3"));
    assert!(rels.is_authorised(
        &ProofPurposeKind::CapabilityInvocation,
        "did:example:alice#key-4"
    ));
    assert!(rels.is_authorised(
        &ProofPurposeKind::CapabilityDelegation,
        "did:example:alice#key-5"
    ));
}

#[test]
fn test_did_relationships_unauthorised() {
    let rels = make_relationships();
    assert!(!rels.is_authorised(&ProofPurposeKind::Authentication, "did:example:alice#key-2"));
    assert!(!rels.is_authorised(
        &ProofPurposeKind::AssertionMethod,
        "did:example:alice#key-1"
    ));
}

#[test]
fn test_did_relationships_methods_for() {
    let rels = make_relationships();
    let auth_methods = rels.methods_for(&ProofPurposeKind::Authentication);
    assert_eq!(auth_methods, vec!["did:example:alice#key-1"]);
}

#[test]
fn test_did_relationships_purposes() {
    let rels = make_relationships();
    let purposes = rels.purposes();
    assert_eq!(purposes.len(), 5);
    assert!(purposes.contains(&"authentication".to_string()));
    assert!(purposes.contains(&"assertionMethod".to_string()));
}

#[test]
fn test_did_relationships_total_entries() {
    let rels = make_relationships();
    assert_eq!(rels.total_entries(), 5);
}

#[test]
fn test_did_relationships_remove() {
    let mut rels = make_relationships();
    assert!(rels.remove(&ProofPurposeKind::Authentication, "did:example:alice#key-1"));
    assert!(!rels.is_authorised(&ProofPurposeKind::Authentication, "did:example:alice#key-1"));
    assert_eq!(rels.total_entries(), 4);
}

#[test]
fn test_did_relationships_remove_nonexistent() {
    let mut rels = make_relationships();
    assert!(!rels.remove(
        &ProofPurposeKind::Authentication,
        "did:example:alice#key-99"
    ));
}

#[test]
fn test_did_relationships_custom_purpose() {
    let mut rels = DidRelationships::new("did:example:carol");
    rels.add_custom("https://custom.example/sign", "did:example:carol#key-c");
    assert!(rels.is_authorised(
        &ProofPurposeKind::Custom("https://custom.example/sign".to_string()),
        "did:example:carol#key-c"
    ));
}

#[test]
fn test_did_relationships_multiple_methods_per_purpose() {
    let mut rels = DidRelationships::new("did:example:dave");
    rels.add_authentication("did:example:dave#key-a");
    rels.add_authentication("did:example:dave#key-b");
    let methods = rels.methods_for(&ProofPurposeKind::Authentication);
    assert_eq!(methods.len(), 2);
}

#[test]
fn test_validate_authentication_success() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();
    let entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
    let r = result.expect("should succeed");
    assert!(r.valid);
    assert_eq!(r.purpose, "authentication");
}

#[test]
fn test_validate_authentication_challenge_mismatch() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();
    let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    entry.challenge = Some("wrong-challenge".to_string());
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::ChallengeMismatch { .. }));
}

#[test]
fn test_validate_authentication_challenge_missing() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();
    let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    entry.challenge = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::ChallengeMissing));
}

#[test]
fn test_validate_authentication_domain_mismatch() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();
    let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    entry.domain = Some("evil.com".to_string());
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::DomainMismatch { .. }));
}

#[test]
fn test_validate_assertion_method_success() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_validate_assertion_method_unauthorised() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-1");
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::Unauthorised { .. }));
}

#[test]
fn test_validate_key_agreement_success() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(ProofPurposeKind::KeyAgreement, "did:example:alice#key-3");
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_validate_key_agreement_unauthorised() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(ProofPurposeKind::KeyAgreement, "did:example:alice#key-99");
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
}

#[test]
fn test_validate_capability_invocation_success() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(
        ProofPurposeKind::CapabilityInvocation,
        "did:example:alice#key-4",
    );
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_validate_capability_invocation_unauthorised() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(
        ProofPurposeKind::CapabilityInvocation,
        "did:example:alice#key-1",
    );
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
}

#[test]
fn test_validate_capability_delegation_success() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(
        ProofPurposeKind::CapabilityDelegation,
        "did:example:alice#key-5",
    );
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_validate_capability_delegation_unauthorised() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(
        ProofPurposeKind::CapabilityDelegation,
        "did:example:alice#key-1",
    );
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
}

#[test]
fn test_validate_expired_proof() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext {
        current_time_ms: 1_800_000_000_000,
        ..Default::default()
    };
    let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
    entry.expires_ms = Some(1_700_000_000_000);
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::Expired { .. }));
}

#[test]
fn test_validate_future_proof() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext {
        current_time_ms: 1_000_000_000_000,
        ..Default::default()
    };
    let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
    entry.created_ms = 1_500_000_000_000;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::FutureProof { .. }));
}

#[test]
fn test_validate_within_clock_skew() {
    let validator = ProofPurposeValidator::new().with_max_clock_skew_ms(600_000);
    let rels = make_relationships();
    let ctx = ValidationContext {
        current_time_ms: 1_700_000_000_000,
        ..Default::default()
    };
    let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
    entry.created_ms = 1_700_000_300_000;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

struct AlwaysAcceptValidator;
impl CustomPurposeValidator for AlwaysAcceptValidator {
    fn validate(
        &self,
        _entry: &ProofEntry,
        _relationships: &DidRelationships,
        _context: &ValidationContext,
    ) -> Result<(), String> {
        Ok(())
    }
}

struct AlwaysRejectValidator;
impl CustomPurposeValidator for AlwaysRejectValidator {
    fn validate(
        &self,
        _entry: &ProofEntry,
        _relationships: &DidRelationships,
        _context: &ValidationContext,
    ) -> Result<(), String> {
        Err("custom rejection".to_string())
    }
}

#[test]
fn test_custom_purpose_accepted() {
    let mut validator = ProofPurposeValidator::new();
    validator.register_custom(
        "https://example.com/custom",
        Box::new(AlwaysAcceptValidator),
    );
    let mut rels = DidRelationships::new("did:example:custom");
    rels.add_custom("https://example.com/custom", "did:example:custom#key-c");

    let entry = ProofEntry {
        purpose: ProofPurposeKind::Custom("https://example.com/custom".to_string()),
        verification_method: "did:example:custom#key-c".to_string(),
        created_ms: 0,
        expires_ms: None,
        challenge: None,
        domain: None,
        nonce: None,
    };
    let ctx = ValidationContext::default();
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_custom_purpose_rejected() {
    let mut validator = ProofPurposeValidator::new();
    validator.register_custom(
        "https://example.com/custom",
        Box::new(AlwaysRejectValidator),
    );
    let mut rels = DidRelationships::new("did:example:custom");
    rels.add_custom("https://example.com/custom", "did:example:custom#key-c");

    let entry = ProofEntry {
        purpose: ProofPurposeKind::Custom("https://example.com/custom".to_string()),
        verification_method: "did:example:custom#key-c".to_string(),
        created_ms: 0,
        expires_ms: None,
        challenge: None,
        domain: None,
        nonce: None,
    };
    let ctx = ValidationContext::default();
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
}

#[test]
fn test_unknown_custom_purpose() {
    let validator = ProofPurposeValidator::new();
    let mut rels = DidRelationships::new("did:example:unknown");
    rels.add_custom("https://unknown.example/purpose", "did:example:unknown#key");

    let entry = ProofEntry {
        purpose: ProofPurposeKind::Custom("https://unknown.example/purpose".to_string()),
        verification_method: "did:example:unknown#key".to_string(),
        created_ms: 0,
        expires_ms: None,
        challenge: None,
        domain: None,
        nonce: None,
    };
    let ctx = ValidationContext::default();
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::UnknownCustomPurpose(_)));
}

#[test]
fn test_registered_custom_purposes() {
    let mut validator = ProofPurposeValidator::new();
    validator.register_custom("urn:alpha", Box::new(AlwaysAcceptValidator));
    validator.register_custom("urn:beta", Box::new(AlwaysAcceptValidator));
    let purposes = validator.registered_custom_purposes();
    assert_eq!(purposes.len(), 2);
    assert!(purposes.contains(&"urn:alpha".to_string()));
    assert!(purposes.contains(&"urn:beta".to_string()));
}

#[test]
fn test_chain_empty() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();
    let result = validator.validate_chain(&[], &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::EmptyChain));
}

#[test]
fn test_chain_single_entry() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();
    let entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    let result = validator.validate_chain(&[entry], &rels, &ctx);
    assert!(result.is_ok());
    let r = result.expect("should succeed");
    assert!(r.valid);
    assert_eq!(r.chain_length, 1);
    assert_eq!(r.entries.len(), 1);
}

#[test]
fn test_chain_multi_step() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();

    let entry1 = ProofEntry {
        purpose: ProofPurposeKind::Authentication,
        verification_method: "did:example:alice#key-1".to_string(),
        created_ms: 1_699_999_000_000,
        expires_ms: Some(1_700_100_000_000),
        challenge: Some("challenge-123".to_string()),
        domain: Some("example.com".to_string()),
        nonce: None,
    };
    let entry2 = ProofEntry {
        purpose: ProofPurposeKind::AssertionMethod,
        verification_method: "did:example:alice#key-2".to_string(),
        created_ms: 1_699_999_500_000,
        expires_ms: None,
        challenge: None,
        domain: Some("example.com".to_string()),
        nonce: None,
    };
    let result = validator.validate_chain(&[entry1, entry2], &rels, &ctx);
    assert!(result.is_ok());
    let r = result.expect("should succeed");
    assert_eq!(r.chain_length, 2);
}

#[test]
fn test_chain_out_of_order() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();

    let entry1 = ProofEntry {
        purpose: ProofPurposeKind::Authentication,
        verification_method: "did:example:alice#key-1".to_string(),
        created_ms: 1_700_000_000_000,
        expires_ms: Some(1_700_100_000_000),
        challenge: Some("challenge-123".to_string()),
        domain: Some("example.com".to_string()),
        nonce: None,
    };
    let entry2 = ProofEntry {
        purpose: ProofPurposeKind::AssertionMethod,
        verification_method: "did:example:alice#key-2".to_string(),
        created_ms: 1_699_998_000_000,
        expires_ms: None,
        challenge: None,
        domain: Some("example.com".to_string()),
        nonce: None,
    };
    let result = validator.validate_chain(&[entry1, entry2], &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::ChainError { index: 1, .. }));
}

#[test]
fn test_chain_with_invalid_entry() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();

    let good = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    let bad = ProofEntry {
        purpose: ProofPurposeKind::AssertionMethod,
        verification_method: "did:example:alice#key-WRONG".to_string(),
        created_ms: 1_700_000_000_000,
        expires_ms: None,
        challenge: None,
        domain: None,
        nonce: None,
    };
    let result = validator.validate_chain(&[good, bad], &rels, &ctx);
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(matches!(err, PurposeError::ChainError { index: 1, .. }));
}

#[test]
fn test_registry_default_has_five_purposes() {
    let reg = ProofPurposeRegistry::new();
    assert_eq!(reg.count(), 5);
}

#[test]
fn test_registry_all_purpose_ids() {
    let reg = ProofPurposeRegistry::new();
    let ids = reg.all_purpose_ids();
    assert!(ids.contains(&"authentication".to_string()));
    assert!(ids.contains(&"assertionMethod".to_string()));
    assert!(ids.contains(&"keyAgreement".to_string()));
    assert!(ids.contains(&"capabilityInvocation".to_string()));
    assert!(ids.contains(&"capabilityDelegation".to_string()));
}

#[test]
fn test_registry_get() {
    let reg = ProofPurposeRegistry::new();
    let auth = reg.get("authentication");
    assert!(auth.is_some());
    let meta = auth.expect("should exist");
    assert_eq!(meta.label, "Authentication");
    assert!(meta.requires_challenge);
    assert!(meta.recommends_domain);
}

#[test]
fn test_registry_get_assertion() {
    let reg = ProofPurposeRegistry::new();
    let am = reg.get("assertionMethod").expect("should exist");
    assert!(!am.requires_challenge);
    assert!(!am.recommends_domain);
}

#[test]
fn test_registry_is_registered() {
    let reg = ProofPurposeRegistry::new();
    assert!(reg.is_registered("keyAgreement"));
    assert!(!reg.is_registered("nonExistentPurpose"));
}

#[test]
fn test_registry_register_custom() {
    let mut reg = ProofPurposeRegistry::new();
    reg.register(PurposeMetadata {
        purpose_id: "https://custom.example/purpose".to_string(),
        label: "Custom Purpose".to_string(),
        description: "A custom proof purpose".to_string(),
        requires_challenge: true,
        recommends_domain: false,
    });
    assert_eq!(reg.count(), 6);
    assert!(reg.is_registered("https://custom.example/purpose"));
}

#[test]
fn test_registry_unregister() {
    let mut reg = ProofPurposeRegistry::new();
    assert!(reg.unregister("authentication"));
    assert_eq!(reg.count(), 4);
    assert!(!reg.is_registered("authentication"));
}

#[test]
fn test_registry_unregister_nonexistent() {
    let mut reg = ProofPurposeRegistry::new();
    assert!(!reg.unregister("nonexistent"));
    assert_eq!(reg.count(), 5);
}

#[test]
fn test_requires_challenge() {
    assert!(ProofPurposeValidator::requires_challenge(
        &ProofPurposeKind::Authentication
    ));
    assert!(!ProofPurposeValidator::requires_challenge(
        &ProofPurposeKind::AssertionMethod
    ));
    assert!(!ProofPurposeValidator::requires_challenge(
        &ProofPurposeKind::KeyAgreement
    ));
}

#[test]
fn test_describe_purpose() {
    let desc = ProofPurposeValidator::describe_purpose(&ProofPurposeKind::Authentication);
    assert!(desc.contains("identity"));
    let desc2 = ProofPurposeValidator::describe_purpose(&ProofPurposeKind::AssertionMethod);
    assert!(desc2.contains("claims"));
}

#[test]
fn test_relationship_name() {
    assert_eq!(
        ProofPurposeValidator::relationship_name(&ProofPurposeKind::Authentication),
        "authentication"
    );
    assert_eq!(
        ProofPurposeValidator::relationship_name(&ProofPurposeKind::CapabilityDelegation),
        "capabilityDelegation"
    );
}

#[test]
fn test_error_display_unauthorised() {
    let err = PurposeError::Unauthorised {
        purpose: "authentication".to_string(),
        verification_method: "did:x#k".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("not authorised"));
}

#[test]
fn test_error_display_expired() {
    let err = PurposeError::Expired {
        expires_ms: 100,
        current_ms: 200,
    };
    let msg = format!("{err}");
    assert!(msg.contains("expired"));
}

#[test]
fn test_error_display_future() {
    let err = PurposeError::FutureProof {
        created_ms: 300,
        current_ms: 100,
    };
    let msg = format!("{err}");
    assert!(msg.contains("future"));
}

#[test]
fn test_error_display_challenge_mismatch() {
    let err = PurposeError::ChallengeMismatch {
        expected: "a".to_string(),
        actual: "b".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("mismatch"));
}

#[test]
fn test_error_display_chain_error() {
    let err = PurposeError::ChainError {
        index: 2,
        reason: "bad".to_string(),
    };
    let msg = format!("{err}");
    assert!(msg.contains("index 2"));
}

#[test]
fn test_error_display_empty_chain() {
    let err = PurposeError::EmptyChain;
    let msg = format!("{err}");
    assert!(msg.contains("at least one"));
}

#[test]
fn test_validate_no_expiration() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_validate_authentication_no_expected_challenge() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext {
        challenge: None,
        domain: None,
        current_time_ms: 1_700_000_000_000,
    };
    let mut entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    entry.challenge = Some("any-challenge".to_string());
    entry.domain = None;
    let result = validator.validate(&entry, &rels, &ctx);
    assert!(result.is_ok());
}

#[test]
fn test_validator_default_construction() {
    let v = ProofPurposeValidator::default();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(ProofPurposeKind::AssertionMethod, "did:example:alice#key-2");
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    assert!(v.validate(&entry, &rels, &ctx).is_ok());
}

#[test]
fn test_validation_result_fields() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = ValidationContext::default();
    let mut entry = make_entry(
        ProofPurposeKind::CapabilityInvocation,
        "did:example:alice#key-4",
    );
    entry.created_ms = 0;
    entry.expires_ms = None;
    entry.challenge = None;
    entry.domain = None;
    let r = validator
        .validate(&entry, &rels, &ctx)
        .expect("should succeed");
    assert!(r.valid);
    assert_eq!(r.purpose, "capabilityInvocation");
    assert_eq!(r.verification_method, "did:example:alice#key-4");
    assert!(r.message.contains("validated successfully"));
}

#[test]
fn test_chain_validation_result_fields() {
    let validator = ProofPurposeValidator::new();
    let rels = make_relationships();
    let ctx = make_context();
    let entry = make_entry(ProofPurposeKind::Authentication, "did:example:alice#key-1");
    let r = validator
        .validate_chain(&[entry], &rels, &ctx)
        .expect("should succeed");
    assert!(r.valid);
    assert_eq!(r.chain_length, 1);
    assert!(r.error.is_none());
    assert_eq!(r.entries.len(), 1);
}
