//! Enhanced SAML 2.0 authentication tests

#![cfg(feature = "saml")]

use chrono::Utc;
use std::collections::HashMap;

use oxirs_fuseki::handlers::saml::{
    determine_authn_strength, meets_minimum_strength, validate_enhanced_saml_assertion,
    validate_mfa_requirements, AuthnStrength, EnhancedSamlConfig, SamlComplianceConfig,
    SamlFederationConfig, SamlMfaRequirement, SamlSessionConfig, ValidatedSamlAssertion,
};

#[cfg(test)]
mod saml_authentication_tests {
    use super::*;

    #[test]
    fn test_authn_strength_determination() {
        // Test low strength contexts
        assert!(matches!(
            determine_authn_strength("urn:oasis:names:tc:SAML:2.0:ac:classes:Password"),
            AuthnStrength::Low
        ));

        assert!(matches!(
            determine_authn_strength(
                "urn:oasis:names:tc:SAML:2.0:ac:classes:PasswordProtectedTransport"
            ),
            AuthnStrength::Low
        ));

        // Test medium strength
        assert!(matches!(
            determine_authn_strength(
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileOneFactorUnregistered"
            ),
            AuthnStrength::Medium
        ));

        // Test high strength
        assert!(matches!(
            determine_authn_strength(
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract"
            ),
            AuthnStrength::High
        ));

        assert!(matches!(
            determine_authn_strength("urn:oasis:names:tc:SAML:2.0:ac:classes:Smartcard"),
            AuthnStrength::High
        ));

        // Test highest strength
        assert!(matches!(
            determine_authn_strength("urn:oasis:names:tc:SAML:2.0:ac:classes:SmartcardPKI"),
            AuthnStrength::Highest
        ));

        // Test unknown context (defaults to low)
        assert!(matches!(
            determine_authn_strength("unknown:context"),
            AuthnStrength::Low
        ));
    }

    #[test]
    fn test_strength_comparison() {
        // Test that higher strengths meet lower requirements
        assert!(meets_minimum_strength(
            &AuthnStrength::High,
            &AuthnStrength::Medium
        ));
        assert!(meets_minimum_strength(
            &AuthnStrength::Highest,
            &AuthnStrength::High
        ));
        assert!(meets_minimum_strength(
            &AuthnStrength::Medium,
            &AuthnStrength::Low
        ));

        // Test equal strengths
        assert!(meets_minimum_strength(
            &AuthnStrength::Medium,
            &AuthnStrength::Medium
        ));

        // Test that lower strengths don't meet higher requirements
        assert!(!meets_minimum_strength(
            &AuthnStrength::Low,
            &AuthnStrength::Medium
        ));
        assert!(!meets_minimum_strength(
            &AuthnStrength::Medium,
            &AuthnStrength::High
        ));
        assert!(!meets_minimum_strength(
            &AuthnStrength::High,
            &AuthnStrength::Highest
        ));
    }

    #[test]
    fn test_mfa_validation_not_required() {
        let assertion = create_test_assertion(Some(
            "urn:oasis:names:tc:SAML:2.0:ac:classes:Password".to_string(),
        ));

        let mfa_config = SamlMfaRequirement {
            required: false,
            accepted_contexts: vec![],
            minimum_strength: AuthnStrength::Low,
            timeout_minutes: 30,
        };

        let result = validate_mfa_requirements(&assertion, &mfa_config);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_mfa_validation_required_and_met() {
        let assertion = create_test_assertion(Some(
            "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract".to_string(),
        ));

        let mfa_config = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec![
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract".to_string(),
            ],
            minimum_strength: AuthnStrength::High,
            timeout_minutes: 30,
        };

        let result = validate_mfa_requirements(&assertion, &mfa_config);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_mfa_validation_context_not_accepted() {
        let assertion = create_test_assertion(Some(
            "urn:oasis:names:tc:SAML:2.0:ac:classes:Password".to_string(),
        ));

        let mfa_config = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec![
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract".to_string(),
            ],
            minimum_strength: AuthnStrength::High,
            timeout_minutes: 30,
        };

        let result = validate_mfa_requirements(&assertion, &mfa_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_mfa_validation_insufficient_strength() {
        let assertion = create_test_assertion(Some(
            "urn:oasis:names:tc:SAML:2.0:ac:classes:Password".to_string(),
        ));

        let mfa_config = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec!["urn:oasis:names:tc:SAML:2.0:ac:classes:Password".to_string()],
            minimum_strength: AuthnStrength::High, // Requires high but assertion is low
            timeout_minutes: 30,
        };

        let result = validate_mfa_requirements(&assertion, &mfa_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_mfa_validation_missing_context() {
        let assertion = create_test_assertion(None); // No authentication context

        let mfa_config = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec![
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract".to_string(),
            ],
            minimum_strength: AuthnStrength::High,
            timeout_minutes: 30,
        };

        let result = validate_mfa_requirements(&assertion, &mfa_config);
        assert!(result.is_err());
    }

    #[test]
    fn test_enhanced_saml_config_structure() {
        let session_config = SamlSessionConfig {
            max_session_duration_hours: 8,
            idle_timeout_minutes: 30,
            concurrent_sessions_allowed: 3,
            session_fixation_protection: true,
            secure_cookie_only: true,
        };

        let federation_config = SamlFederationConfig {
            enable_cross_domain: true,
            trusted_domains: vec!["example.org".to_string(), "partner.com".to_string()],
            metadata_refresh_interval_hours: 24,
            discovery_service_url: Some("https://discovery.example.org".to_string()),
        };

        let compliance_config = SamlComplianceConfig {
            audit_all_assertions: true,
            require_encryption: true,
            minimum_signature_algorithm: "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256"
                .to_string(),
            blacklisted_algorithms: vec!["http://www.w3.org/2000/09/xmldsig#rsa-sha1".to_string()],
            require_destination_validation: true,
        };

        // Test that configurations can be created and have expected values
        assert_eq!(session_config.max_session_duration_hours, 8);
        assert_eq!(federation_config.trusted_domains.len(), 2);
        assert!(compliance_config.audit_all_assertions);
        assert!(compliance_config.require_encryption);
        assert_eq!(compliance_config.blacklisted_algorithms.len(), 1);
    }

    #[test]
    fn test_mfa_requirement_serialization() {
        let mfa_requirement = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec![
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract".to_string(),
                "urn:oasis:names:tc:SAML:2.0:ac:classes:SmartcardPKI".to_string(),
            ],
            minimum_strength: AuthnStrength::High,
            timeout_minutes: 15,
        };

        // Test serialization
        let serialized = serde_json::to_string(&mfa_requirement);
        assert!(serialized.is_ok());

        // Test deserialization
        let deserialized: Result<SamlMfaRequirement, _> =
            serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());

        let restored = deserialized.unwrap();
        assert!(restored.required);
        assert_eq!(restored.accepted_contexts.len(), 2);
        assert!(matches!(restored.minimum_strength, AuthnStrength::High));
        assert_eq!(restored.timeout_minutes, 15);
    }

    #[test]
    fn test_authn_strength_serialization() {
        let strengths = vec![
            AuthnStrength::Low,
            AuthnStrength::Medium,
            AuthnStrength::High,
            AuthnStrength::Highest,
        ];

        for strength in strengths {
            let serialized = serde_json::to_string(&strength);
            assert!(serialized.is_ok());

            let deserialized: Result<AuthnStrength, _> = serde_json::from_str(&serialized.unwrap());
            assert!(deserialized.is_ok());
        }
    }

    #[test]
    fn test_complex_mfa_scenario() {
        // Test a complex MFA scenario with multiple acceptable contexts
        let assertion_high = create_test_assertion(Some(
            "urn:oasis:names:tc:SAML:2.0:ac:classes:Smartcard".to_string(),
        ));

        let assertion_low = create_test_assertion(Some(
            "urn:oasis:names:tc:SAML:2.0:ac:classes:Password".to_string(),
        ));

        let mfa_config = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec![
                "urn:oasis:names:tc:SAML:2.0:ac:classes:Smartcard".to_string(),
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract".to_string(),
                "urn:oasis:names:tc:SAML:2.0:ac:classes:SmartcardPKI".to_string(),
            ],
            minimum_strength: AuthnStrength::High,
            timeout_minutes: 20,
        };

        // High strength assertion should pass
        let result_high = validate_mfa_requirements(&assertion_high, &mfa_config);
        assert!(result_high.is_ok());
        assert!(result_high.unwrap());

        // Low strength assertion should fail
        let result_low = validate_mfa_requirements(&assertion_low, &mfa_config);
        assert!(result_low.is_err());
    }

    #[test]
    fn test_edge_case_empty_contexts() {
        let assertion = create_test_assertion(Some(
            "urn:oasis:names:tc:SAML:2.0:ac:classes:SmartcardPKI".to_string(),
        ));

        let mfa_config = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec![], // No accepted contexts
            minimum_strength: AuthnStrength::Highest,
            timeout_minutes: 30,
        };

        let result = validate_mfa_requirements(&assertion, &mfa_config);
        assert!(result.is_err());
    }

    // Helper function to create test assertions
    fn create_test_assertion(authn_context_class: Option<String>) -> ValidatedSamlAssertion {
        ValidatedSamlAssertion {
            subject: "test@example.org".to_string(),
            issuer: "https://idp.example.org".to_string(),
            attributes: HashMap::new(),
            session_index: "session_123".to_string(),
            not_on_or_after: Utc::now() + chrono::Duration::hours(8),
            audience: "oxirs-fuseki".to_string(),
            assertion_id: "assertion_456".to_string(),
            signature_valid: true,
            conditions_valid: true,
            authn_context_class,
            name_id_format: "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress".to_string(),
            encryption_valid: true,
        }
    }
}

#[cfg(test)]
mod saml_integration_tests {
    use super::*;

    #[test]
    fn test_enterprise_saml_configuration() {
        use oxirs_fuseki::handlers::saml::{SamlIdpConfig, SamlSpConfig};

        // Create a comprehensive enterprise SAML configuration
        let sp_config = SamlSpConfig {
            entity_id: "https://fuseki.enterprise.com".to_string(),
            acs_url: "https://fuseki.enterprise.com/saml/acs".to_string(),
            slo_url: "https://fuseki.enterprise.com/saml/slo".to_string(),
            certificate: Some("SP_CERTIFICATE_DATA".to_string()),
            private_key: Some("SP_PRIVATE_KEY_DATA".to_string()),
            want_assertions_signed: true,
            want_authn_requests_signed: true,
        };

        let mut idp_configs = HashMap::new();
        idp_configs.insert(
            "corporate_idp".to_string(),
            SamlIdpConfig {
                entity_id: "https://corporate.idp.com".to_string(),
                sso_url: "https://corporate.idp.com/sso".to_string(),
                slo_url: Some("https://corporate.idp.com/slo".to_string()),
                certificate: "IDP_CERTIFICATE_DATA".to_string(),
                name_id_format: "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
                    .to_string(),
                attribute_mapping: {
                    let mut mapping = HashMap::new();
                    mapping.insert("email".to_string(), "mail".to_string());
                    mapping.insert("displayName".to_string(), "cn".to_string());
                    mapping.insert("roles".to_string(), "memberOf".to_string());
                    mapping
                },
                signature_required: true,
                encryption_required: true,
            },
        );

        let mfa_requirements = SamlMfaRequirement {
            required: true,
            accepted_contexts: vec![
                "urn:oasis:names:tc:SAML:2.0:ac:classes:MobileTwoFactorContract".to_string(),
                "urn:oasis:names:tc:SAML:2.0:ac:classes:Smartcard".to_string(),
                "urn:oasis:names:tc:SAML:2.0:ac:classes:SmartcardPKI".to_string(),
            ],
            minimum_strength: AuthnStrength::High,
            timeout_minutes: 15,
        };

        let session_config = SamlSessionConfig {
            max_session_duration_hours: 12,
            idle_timeout_minutes: 30,
            concurrent_sessions_allowed: 2,
            session_fixation_protection: true,
            secure_cookie_only: true,
        };

        let federation_config = SamlFederationConfig {
            enable_cross_domain: false, // Enterprise prefers controlled federation
            trusted_domains: vec!["enterprise.com".to_string()],
            metadata_refresh_interval_hours: 6,
            discovery_service_url: None,
        };

        let compliance_config = SamlComplianceConfig {
            audit_all_assertions: true,
            require_encryption: true,
            minimum_signature_algorithm: "http://www.w3.org/2001/04/xmldsig-more#rsa-sha384"
                .to_string(),
            blacklisted_algorithms: vec![
                "http://www.w3.org/2000/09/xmldsig#rsa-sha1".to_string(),
                "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256".to_string(), // Require SHA-384 or better
            ],
            require_destination_validation: true,
        };

        let enhanced_config = EnhancedSamlConfig {
            sp_config,
            idp_configs,
            mfa_requirements,
            session_config,
            federation_config,
            compliance_config,
        };

        // Validate the configuration structure
        assert_eq!(
            enhanced_config.sp_config.entity_id,
            "https://fuseki.enterprise.com"
        );
        assert!(enhanced_config.sp_config.want_assertions_signed);
        assert!(enhanced_config.sp_config.want_authn_requests_signed);

        assert_eq!(enhanced_config.idp_configs.len(), 1);
        assert!(enhanced_config.idp_configs.contains_key("corporate_idp"));

        assert!(enhanced_config.mfa_requirements.required);
        assert_eq!(enhanced_config.mfa_requirements.accepted_contexts.len(), 3);

        assert_eq!(
            enhanced_config.session_config.max_session_duration_hours,
            12
        );
        assert!(enhanced_config.session_config.session_fixation_protection);

        assert!(!enhanced_config.federation_config.enable_cross_domain);
        assert_eq!(enhanced_config.federation_config.trusted_domains.len(), 1);

        assert!(enhanced_config.compliance_config.audit_all_assertions);
        assert!(enhanced_config.compliance_config.require_encryption);
        assert_eq!(
            enhanced_config
                .compliance_config
                .blacklisted_algorithms
                .len(),
            2
        );
    }

    #[test]
    fn test_saml_configuration_serialization() {
        let session_config = SamlSessionConfig {
            max_session_duration_hours: 24,
            idle_timeout_minutes: 60,
            concurrent_sessions_allowed: 5,
            session_fixation_protection: false,
            secure_cookie_only: false,
        };

        // Test serialization to JSON
        let json = serde_json::to_string_pretty(&session_config);
        assert!(json.is_ok());

        let json_str = json.unwrap();
        assert!(json_str.contains("max_session_duration_hours"));
        assert!(json_str.contains("24"));

        // Test deserialization from JSON
        let deserialized: Result<SamlSessionConfig, _> = serde_json::from_str(&json_str);
        assert!(deserialized.is_ok());

        let restored = deserialized.unwrap();
        assert_eq!(restored.max_session_duration_hours, 24);
        assert_eq!(restored.idle_timeout_minutes, 60);
        assert_eq!(restored.concurrent_sessions_allowed, 5);
        assert!(!restored.session_fixation_protection);
        assert!(!restored.secure_cookie_only);
    }
}
