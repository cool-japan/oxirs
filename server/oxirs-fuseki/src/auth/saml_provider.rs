//! [`SamlProvider`] — the SAML 2.0 authentication provider implementation.
//!
//! Split out of the original `saml` module (Round 32 refactor). Drives the
//! SP-initiated and IdP-initiated flows: login URL generation, response
//! processing/validation, user extraction, session management, logout, and SP
//! metadata generation.

use base64::{engine::general_purpose, Engine as _};
use chrono::Utc;
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, SystemTime},
};
use tokio::sync::RwLock;
use url::Url;
use uuid::Uuid;

use crate::{
    auth::{AuthResult, User},
    error::{FusekiError, FusekiResult},
};

use super::saml_helpers::xml_escape;
use super::saml_parser::SamlResponseParser;
use super::saml_types::{
    AuthnRequest, PendingRequest, SamlConfig, SamlProvider, SamlResponse, SamlSession,
};

impl SamlProvider {
    /// Create a new SAML provider
    pub fn new(config: SamlConfig) -> Self {
        Self {
            config,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate login URL (SP-initiated SSO)
    pub async fn generate_login_url(&self, relay_state: Option<String>) -> FusekiResult<Url> {
        let request = AuthnRequest::new(&self.config);
        let request_xml = request.to_xml()?;

        // Store pending request
        let pending = PendingRequest {
            id: request.id.clone(),
            relay_state,
            timestamp: SystemTime::now(),
        };

        let relay_state_clone = pending.relay_state.clone();
        let mut pending_requests = self.pending_requests.write().await;
        pending_requests.insert(request.id.clone(), pending);

        // Encode request for HTTP-Redirect binding
        let encoded = general_purpose::STANDARD.encode(request_xml.as_bytes());

        // Build redirect URL
        let mut url = self.config.idp.sso_url.clone();
        url.query_pairs_mut().append_pair("SAMLRequest", &encoded);

        if let Some(relay) = &relay_state_clone {
            url.query_pairs_mut().append_pair("RelayState", relay);
        }

        Ok(url)
    }

    /// Process SAML response from IdP ACS POST
    pub async fn process_response(
        &self,
        saml_response: &str,
        relay_state: Option<&str>,
    ) -> FusekiResult<User> {
        // Decode base64-encoded SAMLResponse
        let decoded = general_purpose::STANDARD
            .decode(saml_response)
            .map_err(|e| {
                FusekiError::authentication(format!("Failed to decode SAML response: {}", e))
            })?;

        let response_xml = String::from_utf8(decoded).map_err(|e| {
            FusekiError::authentication(format!("Invalid UTF-8 in SAML response: {}", e))
        })?;

        // Parse and validate response
        let response = self.parse_response(&response_xml)?;
        self.validate_response(&response)?;

        // Optionally validate audience against SP entity ID
        for assertion in &response.assertions {
            if !assertion.audiences.is_empty()
                && !assertion.audiences.contains(&self.config.sp.entity_id)
            {
                return Err(FusekiError::authentication(
                    "SAML AudienceRestriction does not include this SP's entity ID",
                ));
            }
        }

        // Extract user information
        let user = self.extract_user_info(&response)?;

        // Create session
        let session = SamlSession {
            user: user.clone(),
            session_index: response
                .assertions
                .first()
                .and_then(|a| a.authn_statement.as_ref())
                .and_then(|s| s.session_index.clone()),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + self.config.session.timeout,
            attributes: self.extract_attributes(&response),
        };

        // Store session
        let session_id = Uuid::new_v4().to_string();
        let mut sessions = self.sessions.write().await;
        sessions.insert(session_id, session);

        // Clean up pending request
        if let Some(in_response_to) = &response.in_response_to {
            let mut pending = self.pending_requests.write().await;
            pending.remove(in_response_to);
        }

        Ok(user)
    }

    /// Parse SAML response XML using `quick-xml`
    fn parse_response(&self, xml: &str) -> FusekiResult<SamlResponse> {
        SamlResponseParser::new(xml, &self.config.idp.certificate).parse()
    }

    /// Validate SAML response status and assertion conditions
    fn validate_response(&self, response: &SamlResponse) -> FusekiResult<()> {
        // Check top-level status
        if response.status.code != "urn:oasis:names:tc:SAML:2.0:status:Success" {
            return Err(FusekiError::authentication(format!(
                "SAML authentication failed with status: {} — {}",
                response.status.code,
                response.status.message.as_deref().unwrap_or("(no message)")
            )));
        }

        if response.assertions.is_empty() {
            return Err(FusekiError::authentication(
                "No assertions in SAML response",
            ));
        }

        let now = Utc::now();

        for assertion in &response.assertions {
            if let Some(conditions) = &assertion.conditions {
                if let Some(not_before) = &conditions.not_before {
                    if now < *not_before {
                        return Err(FusekiError::authentication(format!(
                            "SAML assertion not yet valid (NotBefore={})",
                            not_before.to_rfc3339()
                        )));
                    }
                }

                if let Some(not_after) = &conditions.not_on_or_after {
                    if now >= *not_after {
                        return Err(FusekiError::authentication(format!(
                            "SAML assertion expired (NotOnOrAfter={})",
                            not_after.to_rfc3339()
                        )));
                    }
                }
            }

            // Check SessionNotOnOrAfter if present
            if let Some(stmt) = &assertion.authn_statement {
                if let Some(session_not_after) = &stmt.session_not_on_or_after {
                    if now >= *session_not_after {
                        return Err(FusekiError::authentication(format!(
                            "SAML session expired (SessionNotOnOrAfter={})",
                            session_not_after.to_rfc3339()
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract user information from SAML response assertions
    fn extract_user_info(&self, response: &SamlResponse) -> FusekiResult<User> {
        let assertion = response
            .assertions
            .first()
            .ok_or_else(|| FusekiError::authentication("No assertion found"))?;

        let mut user = User {
            username: assertion.subject.name_id.clone(),
            email: None,
            full_name: None,
            roles: vec!["user".to_string()],
            last_login: Some(chrono::Utc::now()),
            permissions: vec![],
        };

        for attr in &assertion.attributes {
            if attr.name == self.config.attribute_mapping.username && !attr.values.is_empty() {
                user.username = attr.values[0].clone();
            }

            if let Some(email_attr) = &self.config.attribute_mapping.email {
                if attr.name == *email_attr && !attr.values.is_empty() {
                    user.email = Some(attr.values[0].clone());
                }
            }

            if let Some(display_attr) = &self.config.attribute_mapping.display_name {
                if attr.name == *display_attr && !attr.values.is_empty() {
                    user.full_name = Some(attr.values[0].clone());
                }
            }

            if let Some(groups_attr) = &self.config.attribute_mapping.groups {
                if attr.name == *groups_attr {
                    for group in &attr.values {
                        match group.as_str() {
                            "admin" | "administrators" => user.roles.push("admin".to_string()),
                            "editor" | "editors" => user.roles.push("editor".to_string()),
                            _ => {}
                        }
                    }
                }
            }
        }

        Ok(user)
    }

    /// Extract all attributes from all assertions into a flat map
    fn extract_attributes(&self, response: &SamlResponse) -> HashMap<String, Vec<String>> {
        let mut attributes = HashMap::new();

        for assertion in &response.assertions {
            for attr in &assertion.attributes {
                attributes.insert(attr.name.clone(), attr.values.clone());
            }
        }

        attributes
    }

    /// Generate logout URL
    pub async fn generate_logout_url(&self, _user: &User) -> FusekiResult<Option<Url>> {
        Ok(self.config.idp.slo_url.clone())
    }

    /// Clean up expired sessions and stale pending requests
    pub async fn cleanup_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        let now = SystemTime::now();
        sessions.retain(|_, session| session.expires_at > now);

        let mut pending = self.pending_requests.write().await;
        let timeout = Duration::from_secs(300); // 5 minutes

        pending.retain(|_, request| {
            now.duration_since(request.timestamp)
                .unwrap_or(Duration::MAX)
                < timeout
        });
    }

    /// Get session ID by SAML session index
    pub async fn get_session_by_index(&self, session_index: &str) -> FusekiResult<Option<String>> {
        let sessions = self.sessions.read().await;

        for (session_id, session) in sessions.iter() {
            if let Some(index) = &session.session_index {
                if index == session_index {
                    return Ok(Some(session_id.clone()));
                }
            }
        }

        Ok(None)
    }

    /// Generate SAML LogoutRequest XML
    pub async fn generate_logout_request(
        &self,
        session_index: &str,
        name_id: &str,
    ) -> FusekiResult<String> {
        let slo_url = self
            .config
            .idp
            .slo_url
            .as_ref()
            .ok_or_else(|| FusekiError::configuration("SAML SLO not configured"))?;

        let request_id = format!("_{}", Uuid::new_v4());
        let logout_request = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<samlp:LogoutRequest
    xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
    xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
    ID="{}"
    Version="2.0"
    IssueInstant="{}"
    Destination="{}">
    <saml:Issuer>{}</saml:Issuer>
    <saml:NameID>{}</saml:NameID>
    <samlp:SessionIndex>{}</samlp:SessionIndex>
</samlp:LogoutRequest>"#,
            request_id,
            chrono::Utc::now().to_rfc3339(),
            xml_escape(slo_url.as_str()),
            xml_escape(&self.config.sp.entity_id),
            xml_escape(name_id),
            xml_escape(session_index)
        );

        // Encode for HTTP-Redirect binding
        let encoded = general_purpose::STANDARD.encode(logout_request.as_bytes());
        let mut logout_url = slo_url.clone();
        logout_url
            .query_pairs_mut()
            .append_pair("SAMLRequest", &encoded);

        Ok(logout_url.to_string())
    }

    /// Generate SP metadata XML
    pub fn get_metadata(&self) -> String {
        let slo_section = self
            .config
            .sp
            .sls_url
            .as_ref()
            .map(|url| {
                format!(
                    r#"    <md:SingleLogoutService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" Location="{}"/>"#,
                    url
                )
            })
            .unwrap_or_default();

        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata"
                     entityID="{}">
  <md:SPSSODescriptor AuthnRequestsSigned="false" WantAssertionsSigned="true"
                      protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
    <md:NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</md:NameIDFormat>
    <md:AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
                                 Location="{}" index="1" isDefault="true"/>
    {}
  </md:SPSSODescriptor>
</md:EntityDescriptor>"#,
            xml_escape(&self.config.sp.entity_id),
            xml_escape(self.config.sp.acs_url.as_str()),
            slo_section
        )
    }

    /// No-op — SAML does not use username/password authentication
    pub async fn authenticate(&self, _username: &str, _password: &str) -> FusekiResult<AuthResult> {
        Ok(AuthResult::Invalid)
    }

    /// Validate an active SAML session by token (session ID)
    pub async fn validate_token(&self, token: &str) -> FusekiResult<AuthResult> {
        let sessions = self.sessions.read().await;

        if let Some(session) = sessions.get(token) {
            if session.expires_at > SystemTime::now() {
                Ok(AuthResult::Authenticated(session.user.clone()))
            } else {
                Ok(AuthResult::Expired)
            }
        } else {
            Ok(AuthResult::Invalid)
        }
    }

    /// SAML does not support token refresh — callers must re-authenticate.
    pub async fn refresh_token(&self, _token: &str) -> FusekiResult<String> {
        Err(FusekiError::bad_request(
            "SAML does not support token refresh",
        ))
    }

    /// Parse XML directly without base64 decoding.
    /// Exposed for integration tests; callers that need base64 decoding should
    /// use `process_response` instead.
    pub fn parse_response_xml_for_test(&self, xml: &str) -> FusekiResult<SamlResponse> {
        self.parse_response(xml)
    }

    /// Validate a pre-parsed SAML response.
    /// Exposed for integration tests; the full flow uses `process_response`.
    pub fn validate_response_for_test(&self, response: &SamlResponse) -> FusekiResult<()> {
        self.validate_response(response)
    }
}
