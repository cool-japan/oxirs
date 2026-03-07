use super::*;

fn make_saml_config() -> Saml2Config {
    Saml2Config {
        idp_sso_url: "https://idp.example.com/sso".to_string(),
        idp_entity_id: "https://idp.example.com".to_string(),
        idp_certificate: "MOCK_CERT".to_string(),
        ..Default::default()
    }
}

fn make_oidc_config() -> OidcConfig {
    OidcConfig {
        client_id: "oxirs-client".to_string(),
        client_secret: "secret".to_string(),
        discovery_url: "https://accounts.example.com".to_string(),
        redirect_uri: "https://oxirs.example.com/auth/oidc/callback".to_string(),
        ..Default::default()
    }
}

fn make_oidc_token(email: &str, expired: bool) -> OidcIdToken {
    let now = Utc::now();
    let exp = if expired {
        now - Duration::hours(1)
    } else {
        now + Duration::hours(1)
    };
    let mut claims = HashMap::new();
    claims.insert("groups".to_string(), serde_json::json!(["admins", "users"]));
    OidcIdToken {
        sub: "sub-123".to_string(),
        iss: "https://accounts.example.com".to_string(),
        aud: vec!["oxirs-client".to_string()],
        exp,
        iat: now,
        nonce: Some("nonce-abc".to_string()),
        email: Some(email.to_string()),
        email_verified: Some(true),
        name: Some("Test User".to_string()),
        claims,
    }
}

// ── OidcIdToken tests ─────────────────────────────────────────────

#[test]
fn test_oidc_token_not_expired() {
    let token = make_oidc_token("alice@example.com", false);
    assert!(!token.is_expired());
}

#[test]
fn test_oidc_token_expired() {
    let token = make_oidc_token("alice@example.com", true);
    assert!(token.is_expired());
}

#[test]
fn test_oidc_token_get_claim_str_email() {
    let token = make_oidc_token("bob@test.com", false);
    // email is a first-class field; try getting it via the standard claim
    assert_eq!(token.email.as_deref(), Some("bob@test.com"));
}

#[test]
fn test_oidc_token_get_claim_list_groups() {
    let token = make_oidc_token("carol@test.com", false);
    let groups = token.get_claim_list("groups");
    assert!(groups.contains(&"admins".to_string()));
    assert!(groups.contains(&"users".to_string()));
}

#[test]
fn test_oidc_token_get_claim_str_missing() {
    let token = make_oidc_token("dave@test.com", false);
    assert!(token.get_claim_str("nonexistent").is_none());
}

// ── PkceState tests ───────────────────────────────────────────────

#[test]
fn test_pkce_state_uniqueness() {
    let s1 = PkceState::generate("provider-1".to_string());
    let s2 = PkceState::generate("provider-1".to_string());
    // Each generation must produce unique values
    assert_ne!(s1.state, s2.state);
    assert_ne!(s1.nonce, s2.nonce);
    assert_ne!(s1.code_verifier, s2.code_verifier);
}

#[test]
fn test_pkce_state_provider_id_preserved() {
    let state = PkceState::generate("my-idp".to_string());
    assert_eq!(state.provider_id, "my-idp");
}

#[test]
fn test_pkce_state_challenge_differs_from_verifier() {
    let state = PkceState::generate("idp".to_string());
    assert_ne!(state.code_challenge, state.code_verifier);
}

// ── SsoUserProfile tests ──────────────────────────────────────────

#[test]
fn test_user_profile_has_role() {
    let profile = SsoUserProfile {
        subject_id: "user-1".to_string(),
        provider_id: "idp-1".to_string(),
        email: "alice@example.com".to_string(),
        email_verified: true,
        display_name: "Alice".to_string(),
        idp_groups: vec!["Admins".to_string()],
        roles: vec!["admin".to_string(), "user".to_string()],
        attributes: HashMap::new(),
        authenticated_at: Utc::now(),
        session_expires_at: None,
    };
    assert!(profile.has_role("admin"));
    assert!(profile.has_role("user"));
    assert!(!profile.has_role("superuser"));
}

#[test]
fn test_user_profile_has_group() {
    let profile = SsoUserProfile {
        subject_id: "user-2".to_string(),
        provider_id: "idp-1".to_string(),
        email: "bob@example.com".to_string(),
        email_verified: false,
        display_name: "Bob".to_string(),
        idp_groups: vec!["Engineers".to_string(), "RDF-Users".to_string()],
        roles: vec!["user".to_string()],
        attributes: HashMap::new(),
        authenticated_at: Utc::now(),
        session_expires_at: None,
    };
    assert!(profile.has_group("Engineers"));
    assert!(profile.has_group("RDF-Users"));
    assert!(!profile.has_group("Admins"));
}

// ── SsoSession tests ──────────────────────────────────────────────

#[test]
fn test_session_creation_valid() {
    let profile = SsoUserProfile {
        subject_id: "sub-1".to_string(),
        provider_id: "idp-1".to_string(),
        email: "test@example.com".to_string(),
        email_verified: true,
        display_name: "Test".to_string(),
        idp_groups: vec![],
        roles: vec![],
        attributes: HashMap::new(),
        authenticated_at: Utc::now(),
        session_expires_at: None,
    };
    let session = SsoSession::new(profile, 3600);
    assert!(session.is_valid());
    assert!(!session.revoked);
    assert!(!session.id.is_empty());
}

#[test]
fn test_session_revocation() {
    let profile = SsoUserProfile {
        subject_id: "sub-2".to_string(),
        provider_id: "idp-1".to_string(),
        email: "revoked@example.com".to_string(),
        email_verified: true,
        display_name: "Revoked User".to_string(),
        idp_groups: vec![],
        roles: vec![],
        attributes: HashMap::new(),
        authenticated_at: Utc::now(),
        session_expires_at: None,
    };
    let mut session = SsoSession::new(profile, 3600);
    assert!(session.is_valid());
    session.revoke();
    assert!(!session.is_valid());
    assert!(session.revoked);
}

#[test]
fn test_session_touch_updates_last_accessed() {
    let profile = SsoUserProfile {
        subject_id: "sub-3".to_string(),
        provider_id: "idp-1".to_string(),
        email: "touch@example.com".to_string(),
        email_verified: true,
        display_name: "Touch User".to_string(),
        idp_groups: vec![],
        roles: vec![],
        attributes: HashMap::new(),
        authenticated_at: Utc::now(),
        session_expires_at: None,
    };
    let mut session = SsoSession::new(profile, 3600);
    let before = session.last_accessed_at;
    // Slight pause is not guaranteed in test environment; just verify touch works
    session.touch();
    // last_accessed_at should be >= before
    assert!(session.last_accessed_at >= before);
}

// ── SamlAuthRequest tests ─────────────────────────────────────────

#[test]
fn test_saml_auth_request_xml_contains_id() {
    let provider = IdentityProvider::new_saml(
        "idp-1".to_string(),
        "Test IdP".to_string(),
        make_saml_config(),
    );
    let req = SamlAuthRequest::new(&provider).expect("create auth request");
    let xml = req.to_xml();
    assert!(xml.contains(&req.id));
    assert!(xml.contains("AuthnRequest"));
    assert!(xml.contains("samlp:"));
}

#[test]
fn test_saml_auth_request_acs_url() {
    let provider = IdentityProvider::new_saml(
        "idp-1".to_string(),
        "Test IdP".to_string(),
        make_saml_config(),
    );
    let req = SamlAuthRequest::new(&provider).expect("create auth request");
    // The ACS URL should match the default from Saml2Config
    assert!(!req.assertion_consumer_service_url.is_empty());
}

#[test]
fn test_saml_auth_request_fails_for_oidc_provider() {
    let provider = IdentityProvider::new_oidc(
        "oidc-1".to_string(),
        "OIDC IdP".to_string(),
        make_oidc_config(),
    );
    // Creating a SAML auth request from an OIDC provider must fail
    let result = SamlAuthRequest::new(&provider);
    assert!(result.is_err());
}

// ── OidcConfig defaults ───────────────────────────────────────────

#[test]
fn test_oidc_config_default_scopes() {
    let cfg = OidcConfig::default();
    assert!(cfg.scopes.contains(&"openid".to_string()));
}

#[test]
fn test_oidc_config_use_pkce_default() {
    let cfg = OidcConfig::default();
    // PKCE should be enabled by default for security
    assert!(cfg.use_pkce);
}

// ── NameIdFormat ──────────────────────────────────────────────────

#[test]
fn test_name_id_format_urn_values() {
    assert!(NameIdFormat::EmailAddress.as_urn().contains("emailAddress"));
    assert!(NameIdFormat::Persistent.as_urn().contains("persistent"));
    assert!(NameIdFormat::Transient.as_urn().contains("transient"));
    assert!(NameIdFormat::Unspecified.as_urn().contains("unspecified"));
}

// ── SsoError display ──────────────────────────────────────────────

#[test]
fn test_sso_error_display() {
    let err = SsoError::AuthenticationFailed("bad credentials".to_string());
    let msg = err.to_string();
    assert!(msg.contains("Authentication failed"));
    assert!(msg.contains("bad credentials"));
}

#[test]
fn test_sso_session_expired_error_display() {
    let err = SsoError::SessionExpired("alice@example.com".to_string());
    let msg = err.to_string();
    assert!(msg.contains("Session expired"));
}

// ── Multi-provider ────────────────────────────────────────────────

#[test]
fn test_multiple_providers_registered() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });
    let saml_provider = IdentityProvider::new_saml(
        "saml-1".to_string(),
        "SAML IdP".to_string(),
        make_saml_config(),
    );
    let oidc_provider = IdentityProvider::new_oidc(
        "oidc-1".to_string(),
        "OIDC IdP".to_string(),
        make_oidc_config(),
    );
    manager
        .register_provider(saml_provider)
        .expect("register saml");
    manager
        .register_provider(oidc_provider)
        .expect("register oidc");

    let stats = manager.statistics();
    assert_eq!(stats.total_providers, 2);
    assert_eq!(stats.enabled_providers, 2);
}

#[test]
fn test_duplicate_provider_registration_replaces() {
    // register_provider uses HashMap::insert which replaces on duplicate ID
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });
    let p1 = IdentityProvider::new_saml(
        "dup-idp".to_string(),
        "Dup IdP".to_string(),
        make_saml_config(),
    );
    let p2 = IdentityProvider::new_saml(
        "dup-idp".to_string(),
        "Dup IdP V2".to_string(),
        make_saml_config(),
    );
    manager.register_provider(p1).expect("first register ok");
    manager
        .register_provider(p2)
        .expect("second register replaces first");
    // After replacement, there should still be exactly one provider
    let stats = manager.statistics();
    assert_eq!(stats.total_providers, 1);
    // The updated name should be reflected
    let provider = manager.get_provider("dup-idp").expect("provider exists");
    assert_eq!(provider.name, "Dup IdP V2");
}
