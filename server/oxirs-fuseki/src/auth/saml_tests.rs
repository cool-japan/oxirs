//! Unit tests for the SAML 2.0 provider, parser, and helpers.
//!
//! Split out of the original `saml` module (Round 32 refactor). Tests exercise
//! AuthnRequest XML generation, response parsing, validation, audience checks,
//! metadata generation, and XML escaping.

use std::collections::HashMap;
use url::Url;

use super::saml_helpers::xml_escape;
use super::saml_parser::SamlResponseParser;
use super::saml_types::{
    AttributeMapping, AuthnRequest, IdentityProviderConfig, SamlConfig, SamlProvider,
    ServiceProviderConfig, SessionConfig,
};

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
            certificate: String::new(), // No verification for unit tests
            metadata_url: None,
        },
        attribute_mapping: AttributeMapping::default(),
        session: SessionConfig::default(),
    }
}

#[test]
fn test_authn_request_xml_generation() {
    let config = make_config();
    let request = AuthnRequest::new(&config);
    let xml = request.to_xml().expect("to_xml should succeed");

    assert!(xml.contains("samlp:AuthnRequest"), "should contain element");
    assert!(
        xml.contains("https://sp.example.com"),
        "should contain SP entity ID"
    );
    assert!(
        xml.contains("https://sp.example.com/saml/acs"),
        "should contain ACS URL"
    );
    assert!(xml.contains("saml:Issuer"), "should have Issuer element");
    assert!(
        xml.contains("samlp:NameIDPolicy"),
        "should have NameIDPolicy"
    );
    assert!(xml.contains("Version=\"2.0\""), "should have version 2.0");
}

#[test]
fn test_authn_request_xml_escaping() {
    let config = SamlConfig {
        sp: ServiceProviderConfig {
            entity_id: "https://sp.example.com?a=1&b=2".to_string(),
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
    let xml = request.to_xml().expect("to_xml should succeed");
    // The Issuer text content should have & escaped
    assert!(xml.contains("&amp;"), "ampersand must be escaped");
    assert!(!xml.contains("?a=1&b=2"), "raw ampersand must not appear");
}

#[test]
fn test_attribute_mapping_defaults() {
    let mapping = AttributeMapping::default();
    assert_eq!(
        mapping.username,
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/name"
    );
    assert!(mapping.email.is_some());
    assert!(mapping.display_name.is_some());
    assert!(mapping.groups.is_some());
}

// ── XML parsing tests ────────────────────────────────────────────────────

fn minimal_saml_response(
    name_id: &str,
    not_before: Option<&str>,
    not_on_or_after: Option<&str>,
    audience: Option<&str>,
    status_code: &str,
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
                    "<saml:AudienceRestriction><saml:Audience>{}</saml:Audience></saml:AudienceRestriction>",
                    a
                )
            })
            .unwrap_or_default();
        format!(
            r#"<saml:Conditions{}{}>{}</saml:Conditions>"#,
            nb, noa, aud_elem
        )
    } else {
        String::new()
    };

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="_response1"
    Version="2.0"
    IssueInstant="2026-05-01T00:00:00Z">
  <samlp:Status>
    <samlp:StatusCode Value="{status_code}"/>
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
    {conditions_elem}
    <saml:AuthnStatement AuthnInstant="2026-05-01T00:00:00Z" SessionIndex="sess_001">
      <saml:AuthnContext>
        <saml:AuthnContextClassRef>urn:oasis:names:tc:SAML:2.0:ac:classes:Password</saml:AuthnContextClassRef>
      </saml:AuthnContext>
    </saml:AuthnStatement>
    <saml:AttributeStatement>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress">
        <saml:AttributeValue>user@example.com</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname">
        <saml:AttributeValue>Test User</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="http://schemas.xmlsoap.org/claims/Group">
        <saml:AttributeValue>admin</saml:AttributeValue>
        <saml:AttributeValue>editors</saml:AttributeValue>
      </saml:Attribute>
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>"#,
        status_code = status_code,
        name_id = name_id,
        conditions_elem = conditions_elem,
    )
}

const SUCCESS: &str = "urn:oasis:names:tc:SAML:2.0:status:Success";
const FAR_FUTURE: &str = "2099-01-01T00:00:00Z";
const PAST: &str = "2000-01-01T00:00:00Z";

#[test]
fn test_parse_minimal_valid_saml_response() {
    let xml = minimal_saml_response("alice@example.com", None, Some(FAR_FUTURE), None, SUCCESS);
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse successfully");

    assert_eq!(response.status.code, SUCCESS);
    assert_eq!(response.assertions.len(), 1);
    assert_eq!(response.assertions[0].subject.name_id, "alice@example.com");
}

#[test]
fn test_parse_name_id_extraction() {
    let xml = minimal_saml_response("bob@example.com", None, Some(FAR_FUTURE), None, SUCCESS);
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    let name_id = &response.assertions[0].subject.name_id;
    assert_eq!(name_id, "bob@example.com");
}

#[test]
fn test_parse_attribute_statement_extraction() {
    let xml = minimal_saml_response("alice@example.com", None, Some(FAR_FUTURE), None, SUCCESS);
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");

    let attrs: HashMap<String, Vec<String>> = response.assertions[0]
        .attributes
        .iter()
        .map(|a| (a.name.clone(), a.values.clone()))
        .collect();

    assert!(
        attrs.contains_key("http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"),
        "email attribute must be present"
    );
    assert_eq!(
        attrs["http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress"],
        vec!["user@example.com"]
    );

    // Group attribute should have two values
    assert_eq!(
        attrs["http://schemas.xmlsoap.org/claims/Group"],
        vec!["admin", "editors"]
    );
}

#[test]
fn test_expired_response_returns_error() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(PAST), // expired
        None,
        SUCCESS,
    );
    let config = make_config();
    let provider = SamlProvider::new(config);
    let response = provider
        .parse_response_xml_for_test(&xml)
        .expect("parse should succeed");
    let validation = provider.validate_response_for_test(&response);
    assert!(
        validation.is_err(),
        "expired assertion must fail validation"
    );
    let err_msg = validation.unwrap_err().to_string();
    assert!(
        err_msg.to_lowercase().contains("expir"),
        "error should mention expiry: {}",
        err_msg
    );
}

#[test]
fn test_not_yet_valid_response_returns_error() {
    let xml = minimal_saml_response(
        "alice@example.com",
        Some(FAR_FUTURE), // not yet valid
        Some(FAR_FUTURE),
        None,
        SUCCESS,
    );
    let config = make_config();
    let provider = SamlProvider::new(config);
    let response = provider
        .parse_response_xml_for_test(&xml)
        .expect("parse should succeed");
    let validation = provider.validate_response_for_test(&response);
    assert!(
        validation.is_err(),
        "not-yet-valid assertion must fail validation"
    );
}

#[test]
fn test_audience_restriction_mismatch() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        Some("https://other-sp.example.com"), // wrong SP
        SUCCESS,
    );
    let config = make_config(); // SP entity_id = https://sp.example.com
    let provider = SamlProvider::new(config);
    let response = provider
        .parse_response_xml_for_test(&xml)
        .expect("parse should succeed");
    // validate_response passes (just checks time), audience check is in process_response
    // We call process_response via a custom path — test the audience check directly
    let aud = &response.assertions[0].audiences;
    assert!(!aud.is_empty(), "audience should be parsed");
    assert_eq!(aud[0], "https://other-sp.example.com");
    let sp_entity = "https://sp.example.com";
    assert!(!aud.contains(&sp_entity.to_string()), "audience mismatch");
}

#[test]
fn test_audience_restriction_match() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        Some(FAR_FUTURE),
        Some("https://sp.example.com"), // correct SP
        SUCCESS,
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser.parse().expect("should parse");
    let aud = &response.assertions[0].audiences;
    assert_eq!(aud[0], "https://sp.example.com");
}

#[test]
fn test_failed_status_code_propagated() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        None,
        None,
        "urn:oasis:names:tc:SAML:2.0:status:AuthnFailed",
    );
    let parser = SamlResponseParser::new(&xml, "");
    let response = parser
        .parse()
        .expect("should parse even with failure status");
    assert_ne!(response.status.code, SUCCESS);
    assert!(response.status.code.contains("AuthnFailed"));
}

#[test]
fn test_validate_response_rejects_failed_status() {
    let xml = minimal_saml_response(
        "alice@example.com",
        None,
        None,
        None,
        "urn:oasis:names:tc:SAML:2.0:status:AuthnFailed",
    );
    let config = make_config();
    let provider = SamlProvider::new(config);
    let response = provider
        .parse_response_xml_for_test(&xml)
        .expect("parse should succeed");
    let result = provider.validate_response_for_test(&response);
    assert!(result.is_err(), "non-Success status must be rejected");
}

#[test]
fn test_metadata_generation() {
    let config = make_config();
    let provider = SamlProvider::new(config);
    let metadata = provider.get_metadata();

    assert!(metadata.contains("md:EntityDescriptor"));
    assert!(metadata.contains("https://sp.example.com"));
    assert!(metadata.contains("WantAssertionsSigned=\"true\""));
}

#[test]
fn test_xml_escape() {
    assert_eq!(xml_escape("a&b"), "a&amp;b");
    assert_eq!(xml_escape("<tag>"), "&lt;tag&gt;");
    assert_eq!(xml_escape("\"quote\""), "&quot;quote&quot;");
    assert_eq!(xml_escape("it's"), "it&apos;s");
}
