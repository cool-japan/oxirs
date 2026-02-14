//! OAuth2 Provider-Specific Implementations
//!
//! This module provides pre-configured implementations for popular OAuth2/OIDC providers
//! including Google, Microsoft Azure AD, Okta, GitHub, and Auth0.

use crate::config::OAuthConfig;
use serde::{Deserialize, Serialize};

/// Supported OAuth2 providers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OAuth2Provider {
    /// Google OAuth2/OIDC
    Google,
    /// Microsoft Azure Active Directory
    AzureAd,
    /// Okta Identity Platform
    Okta,
    /// GitHub OAuth2
    GitHub,
    /// Auth0 Identity Platform
    Auth0,
    /// Keycloak Identity and Access Management
    Keycloak,
    /// Custom OAuth2 provider
    Custom,
}

/// Provider configuration builder
pub struct ProviderConfigBuilder {
    provider: OAuth2Provider,
    client_id: String,
    client_secret: String,
    tenant_id: Option<String>,      // For Azure AD
    okta_domain: Option<String>,    // For Okta
    auth0_domain: Option<String>,   // For Auth0
    keycloak_realm: Option<String>, // For Keycloak
    keycloak_url: Option<String>,   // For Keycloak
}

impl ProviderConfigBuilder {
    /// Create new provider config builder
    pub fn new(
        provider: OAuth2Provider,
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
    ) -> Self {
        ProviderConfigBuilder {
            provider,
            client_id: client_id.into(),
            client_secret: client_secret.into(),
            tenant_id: None,
            okta_domain: None,
            auth0_domain: None,
            keycloak_realm: None,
            keycloak_url: None,
        }
    }

    /// Set Azure AD tenant ID
    pub fn tenant_id(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Set Okta domain
    pub fn okta_domain(mut self, domain: impl Into<String>) -> Self {
        self.okta_domain = Some(domain.into());
        self
    }

    /// Set Auth0 domain
    pub fn auth0_domain(mut self, domain: impl Into<String>) -> Self {
        self.auth0_domain = Some(domain.into());
        self
    }

    /// Set Keycloak realm
    pub fn keycloak_realm(mut self, realm: impl Into<String>) -> Self {
        self.keycloak_realm = Some(realm.into());
        self
    }

    /// Set Keycloak URL
    pub fn keycloak_url(mut self, url: impl Into<String>) -> Self {
        self.keycloak_url = Some(url.into());
        self
    }

    /// Build OAuth configuration for the provider
    pub fn build(self) -> Result<OAuthConfig, String> {
        match self.provider {
            OAuth2Provider::Google => self.build_google_config(),
            OAuth2Provider::AzureAd => self.build_azure_config(),
            OAuth2Provider::Okta => self.build_okta_config(),
            OAuth2Provider::GitHub => self.build_github_config(),
            OAuth2Provider::Auth0 => self.build_auth0_config(),
            OAuth2Provider::Keycloak => self.build_keycloak_config(),
            OAuth2Provider::Custom => {
                Err("Custom provider must be configured manually".to_string())
            }
        }
    }

    /// Build Google OAuth2/OIDC configuration
    fn build_google_config(self) -> Result<OAuthConfig, String> {
        Ok(OAuthConfig {
            provider: "google".to_string(),
            client_id: self.client_id,
            client_secret: self.client_secret,
            auth_url: "https://accounts.google.com/o/oauth2/v2/auth".to_string(),
            token_url: "https://oauth2.googleapis.com/token".to_string(),
            user_info_url: "https://openidconnect.googleapis.com/v1/userinfo".to_string(),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
        })
    }

    /// Build Azure AD OAuth2/OIDC configuration
    fn build_azure_config(self) -> Result<OAuthConfig, String> {
        let tenant = self
            .tenant_id
            .ok_or_else(|| "Azure AD requires tenant_id".to_string())?;

        Ok(OAuthConfig {
            provider: "azure_ad".to_string(),
            client_id: self.client_id,
            client_secret: self.client_secret,
            auth_url: format!(
                "https://login.microsoftonline.com/{}/oauth2/v2.0/authorize",
                tenant
            ),
            token_url: format!(
                "https://login.microsoftonline.com/{}/oauth2/v2.0/token",
                tenant
            ),
            user_info_url: "https://graph.microsoft.com/v1.0/me".to_string(),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
                "User.Read".to_string(),
            ],
        })
    }

    /// Build Okta OAuth2/OIDC configuration
    fn build_okta_config(self) -> Result<OAuthConfig, String> {
        let domain = self
            .okta_domain
            .ok_or_else(|| "Okta requires okta_domain".to_string())?;

        Ok(OAuthConfig {
            provider: "okta".to_string(),
            client_id: self.client_id,
            client_secret: self.client_secret,
            auth_url: format!("https://{}/oauth2/v1/authorize", domain),
            token_url: format!("https://{}/oauth2/v1/token", domain),
            user_info_url: format!("https://{}/oauth2/v1/userinfo", domain),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
        })
    }

    /// Build GitHub OAuth2 configuration
    fn build_github_config(self) -> Result<OAuthConfig, String> {
        Ok(OAuthConfig {
            provider: "github".to_string(),
            client_id: self.client_id,
            client_secret: self.client_secret,
            auth_url: "https://github.com/login/oauth/authorize".to_string(),
            token_url: "https://github.com/login/oauth/access_token".to_string(),
            user_info_url: "https://api.github.com/user".to_string(),
            scopes: vec!["read:user".to_string(), "user:email".to_string()],
        })
    }

    /// Build Auth0 OAuth2/OIDC configuration
    fn build_auth0_config(self) -> Result<OAuthConfig, String> {
        let domain = self
            .auth0_domain
            .ok_or_else(|| "Auth0 requires auth0_domain".to_string())?;

        Ok(OAuthConfig {
            provider: "auth0".to_string(),
            client_id: self.client_id,
            client_secret: self.client_secret,
            auth_url: format!("https://{}/authorize", domain),
            token_url: format!("https://{}/oauth/token", domain),
            user_info_url: format!("https://{}/userinfo", domain),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
        })
    }

    /// Build Keycloak OAuth2/OIDC configuration
    fn build_keycloak_config(self) -> Result<OAuthConfig, String> {
        let realm = self
            .keycloak_realm
            .ok_or_else(|| "Keycloak requires keycloak_realm".to_string())?;
        let url = self
            .keycloak_url
            .ok_or_else(|| "Keycloak requires keycloak_url".to_string())?;

        let base_url = format!(
            "{}/realms/{}/protocol/openid-connect",
            url.trim_end_matches('/'),
            realm
        );

        Ok(OAuthConfig {
            provider: "keycloak".to_string(),
            client_id: self.client_id,
            client_secret: self.client_secret,
            auth_url: format!("{}/auth", base_url),
            token_url: format!("{}/token", base_url),
            user_info_url: format!("{}/userinfo", base_url),
            scopes: vec![
                "openid".to_string(),
                "profile".to_string(),
                "email".to_string(),
            ],
        })
    }
}

/// Google OAuth2/OIDC provider
pub struct GoogleProvider;

impl GoogleProvider {
    /// Create Google provider configuration
    pub fn config(client_id: impl Into<String>, client_secret: impl Into<String>) -> OAuthConfig {
        ProviderConfigBuilder::new(OAuth2Provider::Google, client_id, client_secret)
            .build()
            .expect("Google OAuth config should be valid")
    }

    /// Default scopes for Google OAuth2
    pub fn default_scopes() -> Vec<String> {
        vec![
            "openid".to_string(),
            "profile".to_string(),
            "email".to_string(),
            "https://www.googleapis.com/auth/userinfo.profile".to_string(),
            "https://www.googleapis.com/auth/userinfo.email".to_string(),
        ]
    }
}

/// Azure AD OAuth2/OIDC provider
pub struct AzureAdProvider;

impl AzureAdProvider {
    /// Create Azure AD provider configuration
    pub fn config(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        tenant_id: impl Into<String>,
    ) -> Result<OAuthConfig, String> {
        ProviderConfigBuilder::new(OAuth2Provider::AzureAd, client_id, client_secret)
            .tenant_id(tenant_id)
            .build()
    }

    /// Default scopes for Azure AD
    pub fn default_scopes() -> Vec<String> {
        vec![
            "openid".to_string(),
            "profile".to_string(),
            "email".to_string(),
            "User.Read".to_string(),
            "offline_access".to_string(),
        ]
    }
}

/// Okta OAuth2/OIDC provider
pub struct OktaProvider;

impl OktaProvider {
    /// Create Okta provider configuration
    pub fn config(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        okta_domain: impl Into<String>,
    ) -> Result<OAuthConfig, String> {
        ProviderConfigBuilder::new(OAuth2Provider::Okta, client_id, client_secret)
            .okta_domain(okta_domain)
            .build()
    }

    /// Default scopes for Okta
    pub fn default_scopes() -> Vec<String> {
        vec![
            "openid".to_string(),
            "profile".to_string(),
            "email".to_string(),
            "groups".to_string(),
        ]
    }
}

/// GitHub OAuth2 provider
pub struct GitHubProvider;

impl GitHubProvider {
    /// Create GitHub provider configuration
    pub fn config(client_id: impl Into<String>, client_secret: impl Into<String>) -> OAuthConfig {
        ProviderConfigBuilder::new(OAuth2Provider::GitHub, client_id, client_secret)
            .build()
            .expect("GitHub OAuth config should be valid")
    }

    /// Default scopes for GitHub OAuth2
    pub fn default_scopes() -> Vec<String> {
        vec![
            "read:user".to_string(),
            "user:email".to_string(),
            "read:org".to_string(),
        ]
    }
}

/// Auth0 OAuth2/OIDC provider
pub struct Auth0Provider;

impl Auth0Provider {
    /// Create Auth0 provider configuration
    pub fn config(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        auth0_domain: impl Into<String>,
    ) -> Result<OAuthConfig, String> {
        ProviderConfigBuilder::new(OAuth2Provider::Auth0, client_id, client_secret)
            .auth0_domain(auth0_domain)
            .build()
    }

    /// Default scopes for Auth0
    pub fn default_scopes() -> Vec<String> {
        vec![
            "openid".to_string(),
            "profile".to_string(),
            "email".to_string(),
        ]
    }
}

/// Keycloak OAuth2/OIDC provider
pub struct KeycloakProvider;

impl KeycloakProvider {
    /// Create Keycloak provider configuration
    pub fn config(
        client_id: impl Into<String>,
        client_secret: impl Into<String>,
        keycloak_url: impl Into<String>,
        realm: impl Into<String>,
    ) -> Result<OAuthConfig, String> {
        ProviderConfigBuilder::new(OAuth2Provider::Keycloak, client_id, client_secret)
            .keycloak_url(keycloak_url)
            .keycloak_realm(realm)
            .build()
    }

    /// Default scopes for Keycloak
    pub fn default_scopes() -> Vec<String> {
        vec![
            "openid".to_string(),
            "profile".to_string(),
            "email".to_string(),
            "roles".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_google_provider_config() {
        let config = GoogleProvider::config("test_client_id", "test_secret");

        assert_eq!(config.provider, "google");
        assert_eq!(config.client_id, "test_client_id");
        assert!(config.auth_url.contains("accounts.google.com"));
        assert!(config.scopes.contains(&"openid".to_string()));
    }

    #[test]
    fn test_azure_ad_provider_config() {
        let config = AzureAdProvider::config("test_client_id", "test_secret", "common").unwrap();

        assert_eq!(config.provider, "azure_ad");
        assert!(config.auth_url.contains("login.microsoftonline.com"));
        assert!(config.scopes.contains(&"User.Read".to_string()));
    }

    #[test]
    fn test_okta_provider_config() {
        let config =
            OktaProvider::config("test_client_id", "test_secret", "dev-123456.okta.com").unwrap();

        assert_eq!(config.provider, "okta");
        assert!(config.auth_url.contains("dev-123456.okta.com"));
        assert!(config.scopes.contains(&"openid".to_string()));
    }

    #[test]
    fn test_github_provider_config() {
        let config = GitHubProvider::config("test_client_id", "test_secret");

        assert_eq!(config.provider, "github");
        assert!(config.auth_url.contains("github.com"));
        assert!(config.scopes.contains(&"read:user".to_string()));
    }

    #[test]
    fn test_auth0_provider_config() {
        let config =
            Auth0Provider::config("test_client_id", "test_secret", "dev-123456.us.auth0.com")
                .unwrap();

        assert_eq!(config.provider, "auth0");
        assert!(config.auth_url.contains("dev-123456.us.auth0.com"));
    }

    #[test]
    fn test_keycloak_provider_config() {
        let config = KeycloakProvider::config(
            "test_client_id",
            "test_secret",
            "https://keycloak.example.com",
            "master",
        )
        .unwrap();

        assert_eq!(config.provider, "keycloak");
        assert!(config.auth_url.contains("keycloak.example.com"));
        assert!(config.auth_url.contains("/realms/master/"));
    }

    #[test]
    fn test_provider_builder() {
        let config = ProviderConfigBuilder::new(OAuth2Provider::Google, "client_id", "secret")
            .build()
            .unwrap();

        assert_eq!(config.provider, "google");
        assert_eq!(config.client_id, "client_id");
    }

    #[test]
    fn test_azure_ad_missing_tenant() {
        let result =
            ProviderConfigBuilder::new(OAuth2Provider::AzureAd, "client_id", "secret").build();

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("tenant_id"));
    }
}
