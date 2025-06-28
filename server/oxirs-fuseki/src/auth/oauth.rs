//! OAuth2/OIDC authentication support for OxiRS Fuseki
//!
//! This module implements OAuth2 and OpenID Connect authentication flows,
//! providing enterprise-grade authentication integration.

use crate::{
    auth::{AuthResult, Permission, User},
    config::OAuthConfig,
    error::{FusekiError, FusekiResult},
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// OAuth2 flow types supported
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OAuth2Flow {
    AuthorizationCode,
    ClientCredentials,
    DeviceCode,
    RefreshToken,
}

/// OAuth2 authentication service
#[derive(Clone)]
pub struct OAuth2Service {
    config: Arc<OAuthConfig>,
    active_states: Arc<RwLock<HashMap<String, OAuth2State>>>,
    access_tokens: Arc<RwLock<HashMap<String, OAuth2Token>>>,
    client: reqwest::Client,
}

/// OAuth2 state for authorization flow
#[derive(Debug, Clone)]
pub struct OAuth2State {
    pub state: String,
    pub code_verifier: Option<String>, // For PKCE
    pub redirect_uri: String,
    pub scopes: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

/// OAuth2 access token information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OAuth2Token {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub refresh_token: Option<String>,
    pub scope: String,
    pub id_token: Option<String>, // For OIDC
    pub issued_at: DateTime<Utc>,
}

/// OIDC user information from userinfo endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OIDCUserInfo {
    pub sub: String,
    pub name: Option<String>,
    pub given_name: Option<String>,
    pub family_name: Option<String>,
    pub email: Option<String>,
    pub email_verified: Option<bool>,
    pub picture: Option<String>,
    pub locale: Option<String>,
    pub groups: Option<Vec<String>>,
    pub roles: Option<Vec<String>>,
}

/// OAuth2 authorization request
#[derive(Debug, Deserialize)]
pub struct OAuth2AuthRequest {
    pub response_type: String,
    pub client_id: String,
    pub redirect_uri: String,
    pub scope: Option<String>,
    pub state: Option<String>,
    pub code_challenge: Option<String>,
    pub code_challenge_method: Option<String>,
}

/// OAuth2 token request
#[derive(Debug, Deserialize)]
pub struct OAuth2TokenRequest {
    pub grant_type: String,
    pub code: Option<String>,
    pub redirect_uri: Option<String>,
    pub client_id: String,
    pub client_secret: Option<String>,
    pub code_verifier: Option<String>,
    pub refresh_token: Option<String>,
}

/// OAuth2 token response
#[derive(Debug, Serialize, Deserialize)]
pub struct OAuth2TokenResponse {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: Option<u64>,
    pub refresh_token: Option<String>,
    pub scope: Option<String>,
    pub id_token: Option<String>,
}

/// OIDC discovery document
#[derive(Debug, Deserialize)]
pub struct OIDCDiscovery {
    pub issuer: String,
    pub authorization_endpoint: String,
    pub token_endpoint: String,
    pub userinfo_endpoint: String,
    pub jwks_uri: String,
    pub scopes_supported: Vec<String>,
    pub response_types_supported: Vec<String>,
    pub grant_types_supported: Vec<String>,
    pub subject_types_supported: Vec<String>,
    pub id_token_signing_alg_values_supported: Vec<String>,
}

/// JWT header for OIDC ID tokens
#[derive(Debug, Deserialize)]
pub struct JWTHeader {
    pub alg: String,
    pub typ: String,
    pub kid: Option<String>,
}

/// OIDC ID token claims
#[derive(Debug, Deserialize)]
pub struct IDTokenClaims {
    pub iss: String,
    pub sub: String,
    pub aud: String,
    pub exp: u64,
    pub iat: u64,
    pub name: Option<String>,
    pub email: Option<String>,
    pub email_verified: Option<bool>,
    pub groups: Option<Vec<String>>,
    pub roles: Option<Vec<String>>,
}

impl OAuth2Service {
    /// Create new OAuth2 service
    pub fn new(config: OAuthConfig) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        OAuth2Service {
            config: Arc::new(config),
            active_states: Arc::new(RwLock::new(HashMap::new())),
            access_tokens: Arc::new(RwLock::new(HashMap::new())),
            client,
        }
    }

    /// Discover OIDC configuration from provider
    pub async fn discover_oidc_config(&self) -> FusekiResult<OIDCDiscovery> {
        let discovery_url = if self
            .config
            .auth_url
            .ends_with("/.well-known/openid_configuration")
        {
            self.config.auth_url.clone()
        } else {
            format!(
                "{}/.well-known/openid_configuration",
                self.config.auth_url.trim_end_matches('/')
            )
        };

        info!("Discovering OIDC configuration from: {}", discovery_url);

        let response = self.client.get(&discovery_url).send().await.map_err(|e| {
            FusekiError::authentication(format!("Failed to fetch OIDC discovery: {}", e))
        })?;

        if !response.status().is_success() {
            return Err(FusekiError::authentication(format!(
                "OIDC discovery failed with status: {}",
                response.status()
            )));
        }

        let discovery: OIDCDiscovery = response.json().await.map_err(|e| {
            FusekiError::authentication(format!("Failed to parse OIDC discovery: {}", e))
        })?;

        debug!("OIDC discovery successful for issuer: {}", discovery.issuer);
        Ok(discovery)
    }

    /// Generate authorization URL for OAuth2 flow
    pub async fn generate_authorization_url(
        &self,
        redirect_uri: &str,
        scopes: &[String],
        use_pkce: bool,
    ) -> FusekiResult<(String, String)> {
        let state = uuid::Uuid::new_v4().to_string();
        let mut url = format!(
            "{}?response_type=code&client_id={}&redirect_uri={}&state={}",
            self.config.auth_url,
            urlencoding::encode(&self.config.client_id),
            urlencoding::encode(redirect_uri),
            urlencoding::encode(&state)
        );

        // Add scopes
        if !scopes.is_empty() {
            let scope_string = scopes.join(" ");
            url.push_str(&format!("&scope={}", urlencoding::encode(&scope_string)));
        } else {
            // Default scopes for OIDC
            url.push_str("&scope=openid%20profile%20email");
        }

        let mut oauth_state = OAuth2State {
            state: state.clone(),
            code_verifier: None,
            redirect_uri: redirect_uri.to_string(),
            scopes: scopes.to_vec(),
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::minutes(10),
        };

        // Add PKCE if requested
        if use_pkce {
            let code_verifier = generate_code_verifier();
            let code_challenge = generate_code_challenge(&code_verifier);

            url.push_str(&format!(
                "&code_challenge={}&code_challenge_method=S256",
                urlencoding::encode(&code_challenge)
            ));

            oauth_state.code_verifier = Some(code_verifier);
        }

        // Store state
        let mut states = self.active_states.write().await;
        states.insert(state.clone(), oauth_state);

        info!("Generated OAuth2 authorization URL for state: {}", state);
        Ok((url, state))
    }

    /// Exchange authorization code for access token
    pub async fn exchange_code_for_token(
        &self,
        code: &str,
        state: &str,
        redirect_uri: &str,
    ) -> FusekiResult<OAuth2Token> {
        // Validate state
        let oauth_state = {
            let mut states = self.active_states.write().await;
            states.remove(state)
        };

        let oauth_state = oauth_state
            .ok_or_else(|| FusekiError::authentication("Invalid or expired OAuth2 state"))?;

        if oauth_state.redirect_uri != redirect_uri {
            return Err(FusekiError::authentication("Redirect URI mismatch"));
        }

        if Utc::now() > oauth_state.expires_at {
            return Err(FusekiError::authentication("OAuth2 state expired"));
        }

        // Prepare token request
        let mut params = vec![
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", redirect_uri),
            ("client_id", &self.config.client_id),
            ("client_secret", &self.config.client_secret),
        ];

        // Add PKCE if used
        if let Some(ref code_verifier) = oauth_state.code_verifier {
            params.push(("code_verifier", code_verifier));
        }

        debug!("Exchanging authorization code for token");

        let response = self
            .client
            .post(&self.config.token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await
            .map_err(|e| FusekiError::authentication(format!("Token exchange failed: {}", e)))?;

        if !response.status().is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(FusekiError::authentication(format!(
                "Token exchange failed with status {}: {}",
                response.status(),
                error_text
            )));
        }

        let token_response: OAuth2TokenResponse = response.json().await.map_err(|e| {
            FusekiError::authentication(format!("Failed to parse token response: {}", e))
        })?;

        let token = OAuth2Token {
            access_token: token_response.access_token,
            token_type: token_response.token_type,
            expires_in: token_response.expires_in.unwrap_or(3600),
            refresh_token: token_response.refresh_token,
            scope: token_response.scope.unwrap_or_default(),
            id_token: token_response.id_token,
            issued_at: Utc::now(),
        };

        // Store token
        let mut tokens = self.access_tokens.write().await;
        tokens.insert(token.access_token.clone(), token.clone());

        info!("Successfully exchanged authorization code for access token");
        Ok(token)
    }

    /// Get user information from OIDC userinfo endpoint
    pub async fn get_user_info(&self, access_token: &str) -> FusekiResult<OIDCUserInfo> {
        debug!("Fetching user info from OIDC userinfo endpoint");

        let response = self
            .client
            .get(&self.config.user_info_url)
            .bearer_auth(access_token)
            .send()
            .await
            .map_err(|e| FusekiError::authentication(format!("UserInfo request failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(FusekiError::authentication(format!(
                "UserInfo request failed with status: {}",
                response.status()
            )));
        }

        let user_info: OIDCUserInfo = response.json().await.map_err(|e| {
            FusekiError::authentication(format!("Failed to parse user info: {}", e))
        })?;

        debug!(
            "Successfully retrieved user info for subject: {}",
            user_info.sub
        );
        Ok(user_info)
    }

    /// Authenticate user using OAuth2/OIDC
    pub async fn authenticate_oauth2(&self, access_token: &str) -> FusekiResult<AuthResult> {
        // Get user info from OIDC provider
        let user_info = self.get_user_info(access_token).await?;

        // Map OIDC user info to internal user
        let user = self.map_oidc_user_to_internal(user_info).await?;

        info!(
            "OAuth2/OIDC authentication successful for user: {}",
            user.username
        );
        Ok(AuthResult::Authenticated(user))
    }

    /// Refresh access token using refresh token
    pub async fn refresh_token(&self, refresh_token: &str) -> FusekiResult<OAuth2Token> {
        debug!("Refreshing OAuth2 access token");

        let params = vec![
            ("grant_type", "refresh_token"),
            ("refresh_token", refresh_token),
            ("client_id", &self.config.client_id),
            ("client_secret", &self.config.client_secret),
        ];

        let response = self
            .client
            .post(&self.config.token_url)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .form(&params)
            .send()
            .await
            .map_err(|e| FusekiError::authentication(format!("Token refresh failed: {}", e)))?;

        if !response.status().is_success() {
            return Err(FusekiError::authentication(format!(
                "Token refresh failed with status: {}",
                response.status()
            )));
        }

        let token_response: OAuth2TokenResponse = response.json().await.map_err(|e| {
            FusekiError::authentication(format!("Failed to parse refresh response: {}", e))
        })?;

        let token = OAuth2Token {
            access_token: token_response.access_token,
            token_type: token_response.token_type,
            expires_in: token_response.expires_in.unwrap_or(3600),
            refresh_token: token_response
                .refresh_token
                .or_else(|| Some(refresh_token.to_string())),
            scope: token_response.scope.unwrap_or_default(),
            id_token: token_response.id_token,
            issued_at: Utc::now(),
        };

        info!("Successfully refreshed OAuth2 access token");
        Ok(token)
    }

    /// Validate access token
    pub async fn validate_access_token(&self, access_token: &str) -> FusekiResult<bool> {
        let tokens = self.access_tokens.read().await;

        if let Some(token) = tokens.get(access_token) {
            let expires_at = token.issued_at + chrono::Duration::seconds(token.expires_in as i64);
            Ok(Utc::now() < expires_at)
        } else {
            // Token not in cache, try to validate with provider
            self.validate_token_with_provider(access_token).await
        }
    }

    /// Validate token with OAuth2 provider introspection endpoint
    async fn validate_token_with_provider(&self, access_token: &str) -> FusekiResult<bool> {
        // Many OAuth2 providers support token introspection (RFC 7662)
        // This is a simplified implementation
        match self.get_user_info(access_token).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Map OIDC user info to internal user structure
    async fn map_oidc_user_to_internal(&self, user_info: OIDCUserInfo) -> FusekiResult<User> {
        // Extract username (prefer email, fallback to subject)
        let username = user_info.email.as_ref().unwrap_or(&user_info.sub).clone();

        // Map groups/roles from OIDC to internal roles
        let mut roles = Vec::new();

        // Add roles from OIDC groups
        if let Some(groups) = &user_info.groups {
            for group in groups {
                roles.push(self.map_oidc_group_to_role(group));
            }
        }

        // Add roles from OIDC roles claim
        if let Some(oidc_roles) = &user_info.roles {
            roles.extend(oidc_roles.clone());
        }

        // Default role if none specified
        if roles.is_empty() {
            roles.push("user".to_string());
        }

        // Compute permissions based on roles
        let permissions = self.compute_permissions_for_roles(&roles).await;

        let user = User {
            username,
            roles,
            email: user_info.email,
            full_name: user_info.name,
            last_login: Some(Utc::now()),
            permissions,
        };

        Ok(user)
    }

    /// Map OIDC group to internal role
    fn map_oidc_group_to_role(&self, group: &str) -> String {
        // Simple mapping - in production this would be configurable
        match group.to_lowercase().as_str() {
            "admin" | "administrators" => "admin".to_string(),
            "writers" | "editors" => "writer".to_string(),
            "readers" | "viewers" => "reader".to_string(),
            _ => "user".to_string(),
        }
    }

    /// Compute permissions for roles
    async fn compute_permissions_for_roles(&self, roles: &[String]) -> Vec<Permission> {
        let mut permissions = Vec::new();

        for role in roles {
            match role.as_str() {
                "admin" => {
                    permissions.extend(vec![
                        Permission::GlobalAdmin,
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                        Permission::UserManagement,
                        Permission::SystemConfig,
                        Permission::SystemMetrics,
                    ]);
                }
                "writer" => {
                    permissions.extend(vec![
                        Permission::GlobalRead,
                        Permission::GlobalWrite,
                        Permission::SparqlQuery,
                        Permission::SparqlUpdate,
                        Permission::GraphStore,
                    ]);
                }
                "reader" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                "user" => {
                    permissions.extend(vec![Permission::GlobalRead, Permission::SparqlQuery]);
                }
                _ => {}
            }
        }

        // Remove duplicates
        permissions.sort();
        permissions.dedup();

        permissions
    }

    /// Cleanup expired states and tokens
    pub async fn cleanup_expired(&self) {
        let now = Utc::now();

        // Cleanup expired states
        {
            let mut states = self.active_states.write().await;
            states.retain(|_, state| state.expires_at > now);
        }

        // Cleanup expired tokens
        {
            let mut tokens = self.access_tokens.write().await;
            tokens.retain(|_, token| {
                let expires_at =
                    token.issued_at + chrono::Duration::seconds(token.expires_in as i64);
                expires_at > now
            });
        }
    }
}

/// Generate code verifier for PKCE
fn generate_code_verifier() -> String {
    use rand::Rng;
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";
    let mut rng = rand::thread_rng();

    (0..128)
        .map(|_| {
            let idx = rng.gen_range(0..CHARSET.len());
            CHARSET[idx] as char
        })
        .collect()
}

/// Generate code challenge for PKCE using S256 method
fn generate_code_challenge(code_verifier: &str) -> String {
    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(code_verifier.as_bytes());
    URL_SAFE_NO_PAD.encode(digest)
}

/// URL encoding helper
mod urlencoding {
    pub fn encode(input: &str) -> String {
        percent_encoding::utf8_percent_encode(input, percent_encoding::NON_ALPHANUMERIC).to_string()
    }
}

use percent_encoding;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OAuthConfig;

    fn create_test_oauth_config() -> OAuthConfig {
        OAuthConfig {
            provider: "test".to_string(),
            client_id: "test_client_id".to_string(),
            client_secret: "test_client_secret".to_string(),
            auth_url: "https://provider.example.com/auth".to_string(),
            token_url: "https://provider.example.com/token".to_string(),
            user_info_url: "https://provider.example.com/userinfo".to_string(),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
        }
    }

    #[tokio::test]
    async fn test_oauth2_service_creation() {
        let config = create_test_oauth_config();
        let service = OAuth2Service::new(config);

        assert_eq!(service.config.provider, "test");
        assert_eq!(service.config.client_id, "test_client_id");
    }

    #[tokio::test]
    async fn test_authorization_url_generation() {
        let config = create_test_oauth_config();
        let service = OAuth2Service::new(config);

        let (url, state) = service
            .generate_authorization_url("http://localhost:3030/callback", &[], false)
            .await
            .unwrap();

        assert!(url.contains("response_type=code"));
        assert!(url.contains("client_id=test_client_id"));
        assert!(url.contains(&state));
        assert!(!state.is_empty());
    }

    #[tokio::test]
    async fn test_pkce_generation() {
        let code_verifier = generate_code_verifier();
        let code_challenge = generate_code_challenge(&code_verifier);

        assert_eq!(code_verifier.len(), 128);
        assert!(!code_challenge.is_empty());
        assert_ne!(code_verifier, code_challenge);
    }

    #[test]
    fn test_oidc_user_mapping() {
        let user_info = OIDCUserInfo {
            sub: "user123".to_string(),
            name: Some("John Doe".to_string()),
            given_name: Some("John".to_string()),
            family_name: Some("Doe".to_string()),
            email: Some("john.doe@example.com".to_string()),
            email_verified: Some(true),
            picture: None,
            locale: Some("en".to_string()),
            groups: Some(vec!["administrators".to_string()]),
            roles: Some(vec!["admin".to_string()]),
        };

        assert_eq!(user_info.sub, "user123");
        assert_eq!(user_info.email.as_ref().unwrap(), "john.doe@example.com");
        assert!(user_info
            .groups
            .as_ref()
            .unwrap()
            .contains(&"administrators".to_string()));
    }

    #[tokio::test]
    async fn test_cleanup_expired() {
        let config = create_test_oauth_config();
        let service = OAuth2Service::new(config);

        // Add expired state
        let expired_state = OAuth2State {
            state: "expired".to_string(),
            code_verifier: None,
            redirect_uri: "http://test".to_string(),
            scopes: vec![],
            created_at: Utc::now() - chrono::Duration::hours(1),
            expires_at: Utc::now() - chrono::Duration::minutes(30),
        };

        {
            let mut states = service.active_states.write().await;
            states.insert("expired".to_string(), expired_state);
        }

        service.cleanup_expired().await;

        let states = service.active_states.read().await;
        assert!(!states.contains_key("expired"));
    }
}
