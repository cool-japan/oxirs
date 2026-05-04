//! SAML 2.0 SP integration tests
//!
//! Tests for the `auth::saml` module XML parsing, validation, and AuthnRequest generation.

#![cfg(feature = "saml")]

use std::collections::HashMap;

use oxirs_fuseki::auth::saml::{
    AttributeMapping, AuthnRequest, IdentityProviderConfig, SamlConfig, SamlProvider,
    SamlResponseParser, ServiceProviderConfig, SessionConfig,
};
use url::Url;

// ── Helpers ──────────────────────────────────────────────────────────────────

fn make_config() -> SamlConfig {
    SamlConfig {
        sp: ServiceProviderConfig {
            entity_id: "https://sp.example.com".to_string(),
            acs_url: Url::parse("https://sp.example.com/saml/acs").expect("url"),
            sls_url: None,
            certificate: None,
            private_key: None,
        },
        idp: IdentityProviderConfig {
            entity_id: "https://idp.example.com".to_string(),
            sso_url: Url::parse("https://idp.example.com/sso").expect("url"),
            slo_url: None,
            // Empty certificate means signature verification is skipped in tests
            certificate: String::new(),
            metadata_url: None,
        },
        attribute_mapping: AttributeMapping::default(),
        session: SessionConfig::default(),
    }
}

/// Construct a minimal well-formed SAMLResponse XML fixture.
fn minimal_saml_response(
    name_id: &str,
    not_before: Option<&str>,
    not_on_or_after: Option<&str>,
    audience: Option<&str>,
    status_code: &str,
    attrs: &[(&str, &[&str])],
) -> String {
    let conditions_elem = if not_before.is_some() || not_on_or_after.is_some() {
        let nb = not_before
            .map(|v| format!(" NotBefore=\"{}\"", v))
            .unwrap_or_default();
        let noa = not_on_or_after
            .map(|v| format!(" NotOnOrAfter=\"{}\"", v))
            .unwrap_or_default();
        let aud_elem = audience
            .map(|a| {
                format!(
                    "<saml:AudienceRestriction>\
                       <saml:Audience>{}</saml:Audience>\
                     </saml:AudienceRestriction>",
                    a
                )
            })
            .unwrap_or_default();
        format!(
            "<saml:Conditions{}{}>{}</saml:Conditions>",
            nb, noa, aud_elem
        )
    } else {
        String::new()
    };

    let attr_xml: String = attrs
        .iter()
        .map(|(name, values)| {
            let vals: String = values
                .iter()
                .map(|v| format!("<saml:AttributeValue>{}</saml:AttributeValue>", v))
                .collect();
            format!(
                "<saml:Attribute Name=\"{}\">{}</saml:Attribute>",
                name, vals
            )
        })
        .collect();

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_response1"
    Version="2.0"
    IssueInstant="2026-05-01T00:00:00Z">
  <samlp:Status>
    <samlp:StatusCode Value="{status}"/>
  </samlp:Status>
  <saml:Assertion
      xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
      ID="_assertion1"
      Version="2.0"
      IssueInstant="2026-05-01T00:00:00Z">
    <saml:Issuer>https://idp.example.com</saml:Issuer>
    <saml:Subject>
      <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">{name_id}</saml:NameID>
    </saml:Subject>
    {conditions}
    <saml:AuthnStatement AuthnInstant="2026-05-01T00:00:00Z" SessionIndex="sess_001">
      <saml:AuthnContext>
        <saml:AuthnContextClassRef>urn:oasis:names:tc:SAML:2.0:ac:classes:Password</saml:AuthnContextClassRef>
      </saml:AuthnContext>
    </saml:AuthnStatement>
    <saml:AttributeStatement>{attr_xml}</saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>"#,
        status = status_code,
        name_id = name_id,
        conditions = conditions_elem,
        attr_xml = attr_xml,
    )
}

const SUCCESS: &str = "urn:oasis:names:tc:SAML:2.0:status:Success";
const FAR_FUTURE: &str = "2099-01-01T00:00:00Z";
const PAST: &str = "2000-01-01T00:00:00Z";

// ── AuthnRequest XML generation tests ────────────────────────────────────────

#[test]
fn test_authn_request_xml_contains_required_elements() {
    let config = make_config();
    let request = AuthnRequest::new(&config);
    let xml = request.to_xml().expect("to_xml must succeed");

    assert!(
        xml.contains("samlp:AuthnRequest"),
        "must have AuthnRequest element"
    );
    assert!(xml.contains("saml:Issuer"), "must have Issuer element");
    assert!(
        xml.contains("samlp:NameIDPolicy"),
        "must have NameIDPolicy element"
    );
    assert!(xml.contains("Version=\"2.0\""), "must declare SAML 2.0");
    assert!(
        xml.contains("urn:oasis:names:tc:SAML:2.0:protocol"),
        "must declare SAML protocol namespace"
    );
    assert!(
        xml.contains("urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"),
        "must declare HTTP-POST binding"
    );
}

#[test]
fn test_authn_request_xml_contains_sp_entity_id() {
    let config = make_config();
    let request = AuthnRequest::new(&config);
    let xml = request.to_xml().expect("to_xml must succeed");

    assert!(
        xml.contains("https://sp.example.com"),
        "must contain SP entity ID"
    );
    assert!(
        xml.contains("https://sp.example.com/saml/acs"),
        "must contain ACS URL"
    );
}

#[test]
fn test_authn_request_xml_destination_matches_idp_sso() {
    let config = make_config();
    let request = AuthnRequest::new(&config);
    let xml = request.to_xml().expect("to_xml must succeed");

    assert!(
        xml.contains("https://idp.example.com/sso"),
        "Destination must be IdP SSO URL"
    );
}

#[test]
fn test_authn_request_xml_escapes_special_characters() {
    let config = SamlConfig {
        sp: ServiceProviderConfig {
            entity_id: "https://sp.example.com?a=1&b=<special>".to_string(),
            acs_url: Url::parse("https://sp.example.com/saml/acs").expect("url"),
            sls_url: None,
            certificate: None,
            private_key: None,
        },
        idp: IdentityProviderConfig {
            entity_id: "https://idp.example.com".to_string(),
            sso_url: Url::parse("https://idp.example.com/sso").expect("url"),
            slo_url: None,
            certificate: String::new(),
            metadata_url: None,
        },
        attribute_mapping: AttributeMapping::default(),
        session: SessionConfig::default(),
    };
    let request = AuthnRequest::new(&config);
    let xml = request.to_xml().expect("to_xml must succeed");

    // Ampersand in entity ID text content must be escaped
    assert!(
        xml.contains("&amp;"),
        "ampersand must be XML-escaped to &amp;"
    );
    // Raw unescaped & must not appear in the Issuer text content
    assert!(
        !xml.contains(">https://sp.example.com?a=1&b="),
        "raw & must not appear in element text"
    );
    // < must be escaped
    assert!(xml.contains("&lt;"), "< must be XML-escaped");
}

// ── SAMLResponse XML parsing tests ───────────────────────────────────────────

#[test]
fn test_parse_minimal_valid_saml_response_succeeds() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        None,
        SUCCESS,
        &[],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse a valid SAMLResponse");

    assert_eq!(response.status.code, SUCCESS);
    assert_eq!(response.assertions.len(), 1);
}

#[test]
fn test_parse_name_id_extraction() {
    let xml = minimal_saml_response(
        "bob@example.com",
        None,
        Some(FAR_FUTURE),
        None,
        SUCCESS,
        &[],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    assert_eq!(response.assertions[0].subject.name_id, "bob@example.com");
    assert_eq!(
        response.assertions[0].subject.format.as_deref(),
        Some("urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress")
    );
}

#[test]
fn test_parse_attribute_statement_extraction() {
    let email_attr = "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress";
    let group_attr = "http://schemas.xmlsoap.org/claims/Group";

    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        None,
        SUCCESS,
        &[
            (email_attr, &["alice@example.com"]),
            (group_attr, &["admin", "editors"]),
        ],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    let assertion = &response.assertions[0];
    let attrs: HashMap<String, Vec<String>> = assertion
        .attributes
        .iter()
        .map(|a| (a.name.clone(), a.values.clone()))
        .collect();

    assert!(
        attrs.contains_key(email_attr),
        "email attribute must be present"
    );
    assert_eq!(attrs[email_attr], vec!["alice@example.com"]);

    let groups = attrs.get(group_attr).expect("group attribute must exist");
    assert!(groups.contains(&"admin".to_string()), "admin group missing");
    assert!(
        groups.contains(&"editors".to_string()),
        "editors group missing"
    );
}

#[test]
fn test_parse_conditions_not_on_or_after_extracted() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        None,
        SUCCESS,
        &[],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    let conditions = response.assertions[0]
        .conditions
        .as_ref()
        .expect("Conditions must be parsed");
    assert!(
        conditions.not_on_or_after.is_some(),
        "NotOnOrAfter must be parsed"
    );
}

#[test]
fn test_parse_audience_restriction_extracted() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        Some("https://sp.example.com"),
        SUCCESS,
        &[],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    let audiences = &response.assertions[0].audiences;
    assert_eq!(audiences.len(), 1);
    assert_eq!(audiences[0], "https://sp.example.com");
}

// ── Validation tests ─────────────────────────────────────────────────────────

#[test]
fn test_expired_assertion_returns_error() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(PAST), // already expired
        None,
        SUCCESS,
        &[],
    );
    let config = make_config();
    let provider = SamlProvider::new(config);
    let response = provider
        .parse_response_xml_for_test(&xml)
        .expect("parsing should succeed");
    let validation = provider.validate_response_for_test(&response);

    assert!(
        validation.is_err(),
        "expired assertion must fail validation"
    );
    let err = validation.unwrap_err().to_string();
    assert!(
        err.to_lowercase().contains("expir")
            || err.to_lowercase().contains("not_on_or_after")
            || err.to_lowercase().contains("NotOnOrAfter"),
        "error must describe expiry: {}",
        err
    );
}

#[test]
fn test_not_yet_valid_assertion_returns_error() {
    let xml = minimal_saml_response(
        "alice@example.com",
        Some(FAR_FUTURE), // not yet valid
        Some(FAR_FUTURE),
        None,
        SUCCESS,
        &[],
    );
    let config = make_config();
    let provider = SamlProvider::new(config);
    let response = provider
        .parse_response_xml_for_test(&xml)
        .expect("parsing should succeed");
    let validation = provider.validate_response_for_test(&response);

    assert!(
        validation.is_err(),
        "not-yet-valid assertion must fail validation"
    );
}

#[test]
fn test_audience_restriction_mismatch_detected() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        Some("https://other-sp.example.com"), // wrong SP
        SUCCESS,
        &[],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    // The provider SP entity ID is "https://sp.example.com"
    let audiences = &response.assertions[0].audiences;
    assert!(!audiences.is_empty(), "audiences must be parsed");
    assert_eq!(audiences[0], "https://other-sp.example.com");
    // Verify the mismatch with the SP entity ID
    assert!(
        !audiences.contains(&"https://sp.example.com".to_string()),
        "audience must not contain SP entity ID"
    );
}

#[test]
fn test_audience_restriction_match_passes() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        Some("https://sp.example.com"),
        SUCCESS,
        &[],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    let audiences = &response.assertions[0].audiences;
    assert!(
        audiences.contains(&"https://sp.example.com".to_string()),
        "SP entity ID must be in audiences"
    );
}

#[test]
fn test_failed_status_code_propagated_to_response() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        None,
        None,
        "urn:oasis:names:tc:SAML:2.0:status:AuthnFailed",
        &[],
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser
        .parse()
        .expect("should parse even with failure status");

    assert_ne!(response.status.code, SUCCESS);
    assert!(
        response.status.code.contains("AuthnFailed"),
        "AuthnFailed must be in status code"
    );
}

#[test]
fn test_validate_response_rejects_failed_status() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        None,
        None,
        "urn:oasis:names:tc:SAML:2.0:status:AuthnFailed",
        &[],
    );
    let config = make_config();
    let provider = SamlProvider::new(config);
    let response = provider
        .parse_response_xml_for_test(&xml)
        .expect("parse should succeed");
    let result = provider.validate_response_for_test(&response);

    assert!(result.is_err(), "non-Success status must be rejected");
}

// ── AuthnRequest XML structure test ──────────────────────────────────────────

#[test]
fn test_authn_request_force_authn_false_default() {
    let config = make_config();
    let request = AuthnRequest::new(&config);
    assert!(!request.force_authn, "ForceAuthn defaults to false");
    let xml = request.to_xml().expect("to_xml must succeed");
    assert!(
        xml.contains("ForceAuthn=\"false\""),
        "ForceAuthn=false must appear"
    );
}

#[test]
fn test_authn_request_id_is_prefixed_with_underscore() {
    let config = make_config();
    let request = AuthnRequest::new(&config);
    assert!(
        request.id.starts_with('_'),
        "SAML request ID must start with underscore per spec"
    );
}

#[test]
fn test_authn_request_unique_ids() {
    let config = make_config();
    let r1 = AuthnRequest::new(&config);
    let r2 = AuthnRequest::new(&config);
    assert_ne!(r1.id, r2.id, "each request must have a unique ID");
}
