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

fn make_assertion(email: &str) -> SamlAssertion {
    let now = Utc::now();
    let mut attributes = HashMap::new();
    attributes.insert(
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress".to_string(),
        vec![email.to_string()],
    );
    attributes.insert(
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/displayname".to_string(),
        vec!["Test User".to_string()],
    );

    SamlAssertion {
        id: format!("_{}", Uuid::new_v4().simple()),
        name_id: email.to_string(),
        name_id_format: NameIdFormat::EmailAddress.as_urn().to_string(),
        issue_instant: now,
        not_before: Some(now - Duration::minutes(5)),
        not_on_or_after: Some(now + Duration::hours(1)),
        session_index: Some("session-123".to_string()),
        issuer: "https://idp.example.com".to_string(),
        attributes,
    }
}

#[test]
fn test_provider_registration() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let provider = IdentityProvider::new_saml(
        "test-idp".to_string(),
        "Test IdP".to_string(),
        make_saml_config(),
    );
    manager
        .register_provider(provider)
        .expect("register provider");

    assert_eq!(manager.list_providers().len(), 1);
    let retrieved = manager.get_provider("test-idp").expect("get provider");
    assert_eq!(retrieved.name, "Test IdP");
    assert_eq!(retrieved.protocol, SsoProtocol::Saml2);
}

#[test]
fn test_saml_flow_initiation() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let provider = IdentityProvider::new_saml(
        "saml-idp".to_string(),
        "SAML IdP".to_string(),
        make_saml_config(),
    );
    manager.register_provider(provider).expect("register");

    let request = manager
        .initiate_saml_flow("saml-idp")
        .expect("initiate SAML");
    assert!(!request.id.is_empty());
    assert_eq!(request.destination, "https://idp.example.com/sso");
}

#[test]
fn test_saml_xml_generation() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let provider = IdentityProvider::new_saml(
        "saml-idp".to_string(),
        "SAML IdP".to_string(),
        make_saml_config(),
    );
    manager.register_provider(provider).expect("register");

    let request = manager.initiate_saml_flow("saml-idp").expect("initiate");
    let xml = request.to_xml();
    assert!(xml.contains("AuthnRequest"));
    assert!(xml.contains("samlp:AuthnRequest"));
    assert!(xml.contains("https://idp.example.com/sso"));
}

#[test]
fn test_saml_response_processing() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let mut provider = IdentityProvider::new_saml(
        "saml-idp".to_string(),
        "SAML IdP".to_string(),
        make_saml_config(),
    );
    provider.domains = vec!["example.com".to_string()];
    manager.register_provider(provider).expect("register");

    let assertion = make_assertion("alice@example.com");
    let session = manager
        .process_saml_response("saml-idp", assertion)
        .expect("process SAML response");

    assert_eq!(session.user_profile.email, "alice@example.com");
    assert_eq!(session.user_profile.display_name, "Test User");
    assert!(session.is_valid());
}

#[test]
fn test_session_revocation() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let provider = IdentityProvider::new_saml(
        "saml-idp".to_string(),
        "SAML IdP".to_string(),
        make_saml_config(),
    );
    manager.register_provider(provider).expect("register");

    let assertion = make_assertion("bob@test.com");
    let session = manager
        .process_saml_response("saml-idp", assertion)
        .expect("process");

    let session_id = session.id.clone();
    assert_eq!(manager.active_session_count(), 1);

    manager.revoke_session(&session_id).expect("revoke");
    assert_eq!(manager.active_session_count(), 0);
}

#[test]
fn test_oidc_flow_initiation() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let provider = IdentityProvider::new_oidc(
        "oidc-idp".to_string(),
        "OIDC IdP".to_string(),
        make_oidc_config(),
    );
    manager.register_provider(provider).expect("register");

    let auth_request = manager
        .initiate_oidc_flow("oidc-idp")
        .expect("initiate OIDC");
    assert!(!auth_request.state.is_empty());
    assert!(!auth_request.authorization_url.is_empty());
    assert!(auth_request
        .authorization_url
        .contains("response_type=code"));
}

#[test]
fn test_domain_matching() {
    let mut provider =
        IdentityProvider::new_saml("test".to_string(), "Test".to_string(), make_saml_config());
    provider.domains = vec!["acme.com".to_string(), "acme.org".to_string()];

    assert!(provider.handles_domain("alice@acme.com"));
    assert!(provider.handles_domain("bob@acme.org"));
    assert!(!provider.handles_domain("carol@other.com"));
}

#[test]
fn test_find_provider_for_email() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let mut provider = IdentityProvider::new_saml(
        "acme-idp".to_string(),
        "Acme IdP".to_string(),
        make_saml_config(),
    );
    provider.domains = vec!["acme.com".to_string()];
    manager.register_provider(provider).expect("register");

    let found = manager.find_provider_for_email("alice@acme.com");
    assert!(found.is_some());
    assert_eq!(found.expect("found provider").id, "acme-idp");

    let not_found = manager.find_provider_for_email("alice@other.com");
    assert!(not_found.is_none());
}

#[test]
fn test_assertion_validity_window() {
    let now = Utc::now();
    let valid_assertion = SamlAssertion {
        id: "test".to_string(),
        name_id: "user@test.com".to_string(),
        name_id_format: NameIdFormat::EmailAddress.as_urn().to_string(),
        issue_instant: now,
        not_before: Some(now - Duration::minutes(5)),
        not_on_or_after: Some(now + Duration::hours(1)),
        session_index: None,
        issuer: "test-issuer".to_string(),
        attributes: HashMap::new(),
    };
    assert!(valid_assertion.is_valid(300));

    let expired_assertion = SamlAssertion {
        not_on_or_after: Some(now - Duration::minutes(1)),
        ..valid_assertion.clone()
    };
    assert!(!expired_assertion.is_valid(0));
}

#[test]
fn test_sso_statistics() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        ..Default::default()
    });

    let provider = IdentityProvider::new_saml(
        "test-idp".to_string(),
        "Test IdP".to_string(),
        make_saml_config(),
    );
    manager.register_provider(provider).expect("register");

    let assertion = make_assertion("user@example.com");
    let _session = manager
        .process_saml_response("test-idp", assertion)
        .expect("process");

    let stats = manager.statistics();
    assert_eq!(stats.total_providers, 1);
    assert_eq!(stats.enabled_providers, 1);
    assert_eq!(stats.active_sessions, 1);
    assert!(stats.login_successes > 0);
}

#[test]
fn test_cleanup_expired_sessions() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        session_ttl_seconds: 1,
        ..Default::default()
    });

    let provider = IdentityProvider::new_saml(
        "test-idp".to_string(),
        "Test IdP".to_string(),
        make_saml_config(),
    );
    manager.register_provider(provider).expect("register");

    // Create a session that is already expired
    let assertion = make_assertion("user@expire.com");
    let session = manager
        .process_saml_response("test-idp", assertion)
        .expect("process");

    // Manually mark it as expired by modifying
    if let Some(s) = manager.sessions.get_mut(&session.id) {
        s.expires_at = Utc::now() - Duration::seconds(1);
    }

    let cleaned = manager.cleanup_expired_sessions();
    assert_eq!(cleaned, 1);
    assert_eq!(manager.active_session_count(), 0);
}

#[test]
fn test_pkce_state_generation() {
    let state = PkceState::generate("test-provider".to_string());
    assert!(!state.code_verifier.is_empty());
    assert!(!state.code_challenge.is_empty());
    assert!(!state.state.is_empty());
    assert!(!state.nonce.is_empty());
    assert!(!state.is_expired());
}

#[test]
fn test_role_mapping() {
    let mut provider =
        IdentityProvider::new_saml("test".to_string(), "Test".to_string(), make_saml_config());
    provider
        .role_mapping
        .insert("Admins".to_string(), "admin".to_string());
    provider
        .role_mapping
        .insert("Users".to_string(), "user".to_string());

    assert_eq!(provider.map_role("Admins"), "admin");
    assert_eq!(provider.map_role("Users"), "user");
    assert_eq!(provider.map_role("Unknown"), "user"); // falls back to default_role
}

#[test]
fn test_audit_log() {
    let mut manager = SsoAuthManager::new(SsoConfig {
        enabled: true,
        enable_audit_log: true,
        audit_log_buffer_size: 100,
        ..Default::default()
    });

    let provider = IdentityProvider::new_saml(
        "test-idp".to_string(),
        "Test".to_string(),
        make_saml_config(),
    );
    manager.register_provider(provider).expect("register");

    // One event should have been logged for ProviderRegistered
    assert!(!manager.recent_audit_events(10).is_empty());
}
