//! OAuth2 Token Introspection and Revocation
//!
//! This module implements:
//! - RFC 7662: OAuth 2.0 Token Introspection
//! - RFC 7009: OAuth 2.0 Token Revocation
//! - Token lifecycle management
//! - Revoked token tracking

use crate::error::{FusekiError, FusekiResult};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Token introspection response (RFC 7662)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenIntrospectionResponse {
    /// Token is active
    pub active: bool,
    /// Token scope
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
    /// Client ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
    /// Username
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
    /// Token type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_type: Option<String>,
    /// Expiration time (Unix timestamp)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exp: Option<i64>,
    /// Issued at time (Unix timestamp)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iat: Option<i64>,
    /// Not before time (Unix timestamp)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nbf: Option<i64>,
    /// Subject
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sub: Option<String>,
    /// Audience
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aud: Option<String>,
    /// Issuer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iss: Option<String>,
    /// JWT ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jti: Option<String>,
}

/// Token revocation request (RFC 7009)
#[derive(Debug, Deserialize)]
pub struct TokenRevocationRequest {
    /// Token to revoke
    pub token: String,
    /// Token type hint (access_token or refresh_token)
    pub token_type_hint: Option<String>,
    /// Client ID
    pub client_id: Option<String>,
    /// Client secret
    pub client_secret: Option<String>,
}

/// Token information stored in the token manager
#[derive(Debug, Clone)]
pub struct TokenInfo {
    /// Token value
    pub token: String,
    /// Token type (access_token, refresh_token, id_token)
    pub token_type: String,
    /// Client ID
    pub client_id: String,
    /// Username/Subject
    pub subject: String,
    /// Scope
    pub scope: String,
    /// Issued at
    pub issued_at: DateTime<Utc>,
    /// Expires at
    pub expires_at: DateTime<Utc>,
    /// Audience
    pub audience: Option<String>,
    /// Issuer
    pub issuer: Option<String>,
    /// JWT ID
    pub jti: Option<String>,
}

/// Token manager for introspection and revocation
pub struct TokenManager {
    /// Active tokens by token value
    active_tokens: Arc<RwLock<HashMap<String, TokenInfo>>>,
    /// Revoked tokens (token value)
    revoked_tokens: Arc<RwLock<HashSet<String>>>,
    /// Revoked tokens by JWT ID
    revoked_jti: Arc<RwLock<HashSet<String>>>,
    /// Token expiration cleanup interval
    cleanup_interval_secs: u64,
}

impl TokenManager {
    /// Create new token manager
    pub fn new(cleanup_interval_secs: u64) -> Self {
        TokenManager {
            active_tokens: Arc::new(RwLock::new(HashMap::new())),
            revoked_tokens: Arc::new(RwLock::new(HashSet::new())),
            revoked_jti: Arc::new(RwLock::new(HashSet::new())),
            cleanup_interval_secs,
        }
    }

    /// Register a new token
    pub async fn register_token(&self, token_info: TokenInfo) -> FusekiResult<()> {
        let mut tokens = self.active_tokens.write().await;
        tokens.insert(token_info.token.clone(), token_info);
        debug!("Registered new token");
        Ok(())
    }

    /// Introspect a token (RFC 7662)
    pub async fn introspect_token(
        &self,
        token: &str,
        _client_id: Option<&str>,
    ) -> FusekiResult<TokenIntrospectionResponse> {
        // Check if token is revoked
        {
            let revoked = self.revoked_tokens.read().await;
            if revoked.contains(token) {
                debug!("Token introspection: token is revoked");
                return Ok(TokenIntrospectionResponse {
                    active: false,
                    scope: None,
                    client_id: None,
                    username: None,
                    token_type: None,
                    exp: None,
                    iat: None,
                    nbf: None,
                    sub: None,
                    aud: None,
                    iss: None,
                    jti: None,
                });
            }
        }

        // Get token info
        let tokens = self.active_tokens.read().await;
        if let Some(token_info) = tokens.get(token) {
            // Check expiration
            if Utc::now() > token_info.expires_at {
                debug!("Token introspection: token has expired");
                return Ok(TokenIntrospectionResponse {
                    active: false,
                    scope: None,
                    client_id: None,
                    username: None,
                    token_type: None,
                    exp: None,
                    iat: None,
                    nbf: None,
                    sub: None,
                    aud: None,
                    iss: None,
                    jti: None,
                });
            }

            // Check JTI revocation
            if let Some(ref jti) = token_info.jti {
                let revoked_jti = self.revoked_jti.read().await;
                if revoked_jti.contains(jti) {
                    debug!("Token introspection: token JTI is revoked");
                    return Ok(TokenIntrospectionResponse {
                        active: false,
                        scope: None,
                        client_id: None,
                        username: None,
                        token_type: None,
                        exp: None,
                        iat: None,
                        nbf: None,
                        sub: None,
                        aud: None,
                        iss: None,
                        jti: None,
                    });
                }
            }

            // Token is active
            debug!("Token introspection: token is active");
            Ok(TokenIntrospectionResponse {
                active: true,
                scope: Some(token_info.scope.clone()),
                client_id: Some(token_info.client_id.clone()),
                username: Some(token_info.subject.clone()),
                token_type: Some(token_info.token_type.clone()),
                exp: Some(token_info.expires_at.timestamp()),
                iat: Some(token_info.issued_at.timestamp()),
                nbf: None,
                sub: Some(token_info.subject.clone()),
                aud: token_info.audience.clone(),
                iss: token_info.issuer.clone(),
                jti: token_info.jti.clone(),
            })
        } else {
            // Token not found
            debug!("Token introspection: token not found");
            Ok(TokenIntrospectionResponse {
                active: false,
                scope: None,
                client_id: None,
                username: None,
                token_type: None,
                exp: None,
                iat: None,
                nbf: None,
                sub: None,
                aud: None,
                iss: None,
                jti: None,
            })
        }
    }

    /// Revoke a token (RFC 7009)
    pub async fn revoke_token(&self, request: TokenRevocationRequest) -> FusekiResult<()> {
        info!("Revoking token (type hint: {:?})", request.token_type_hint);

        // Add to revoked tokens set
        {
            let mut revoked = self.revoked_tokens.write().await;
            revoked.insert(request.token.clone());
        }

        // If we have the token info, also revoke by JTI
        {
            let tokens = self.active_tokens.read().await;
            if let Some(token_info) = tokens.get(&request.token) {
                if let Some(ref jti) = token_info.jti {
                    let mut revoked_jti = self.revoked_jti.write().await;
                    revoked_jti.insert(jti.clone());
                    debug!("Also revoked token by JTI: {}", jti);
                }
            }
        }

        // Remove from active tokens
        {
            let mut tokens = self.active_tokens.write().await;
            tokens.remove(&request.token);
        }

        info!("Token revoked successfully");
        Ok(())
    }

    /// Revoke all tokens for a client
    pub async fn revoke_all_client_tokens(&self, client_id: &str) -> FusekiResult<usize> {
        let mut count = 0;

        // Find all tokens for this client
        let tokens_to_revoke: Vec<String> = {
            let tokens = self.active_tokens.read().await;
            tokens
                .values()
                .filter(|info| info.client_id == client_id)
                .map(|info| info.token.clone())
                .collect()
        };

        // Revoke each token
        for token in tokens_to_revoke {
            self.revoke_token(TokenRevocationRequest {
                token,
                token_type_hint: None,
                client_id: Some(client_id.to_string()),
                client_secret: None,
            })
            .await?;
            count += 1;
        }

        info!("Revoked {} tokens for client: {}", count, client_id);
        Ok(count)
    }

    /// Revoke all tokens for a user/subject
    pub async fn revoke_all_user_tokens(&self, subject: &str) -> FusekiResult<usize> {
        let mut count = 0;

        // Find all tokens for this subject
        let tokens_to_revoke: Vec<String> = {
            let tokens = self.active_tokens.read().await;
            tokens
                .values()
                .filter(|info| info.subject == subject)
                .map(|info| info.token.clone())
                .collect()
        };

        // Revoke each token
        for token in tokens_to_revoke {
            self.revoke_token(TokenRevocationRequest {
                token,
                token_type_hint: None,
                client_id: None,
                client_secret: None,
            })
            .await?;
            count += 1;
        }

        info!("Revoked {} tokens for user: {}", count, subject);
        Ok(count)
    }

    /// Check if a token is revoked
    pub async fn is_token_revoked(&self, token: &str) -> bool {
        let revoked = self.revoked_tokens.read().await;
        revoked.contains(token)
    }

    /// Check if a JTI is revoked
    pub async fn is_jti_revoked(&self, jti: &str) -> bool {
        let revoked_jti = self.revoked_jti.read().await;
        revoked_jti.contains(jti)
    }

    /// Cleanup expired tokens and old revocations
    pub async fn cleanup_expired(&self) {
        let now = Utc::now();

        // Remove expired active tokens
        let removed_active = {
            let mut tokens = self.active_tokens.write().await;
            let before_count = tokens.len();
            tokens.retain(|_, info| info.expires_at > now);
            before_count - tokens.len()
        };

        // Keep revoked tokens for a while (e.g., 30 days) to prevent replay
        // In production, you might want to persist revoked tokens to prevent
        // replay attacks even after server restart
        let _revocation_expiry = now - chrono::Duration::days(30);

        // For now, we'll just log the cleanup
        if removed_active > 0 {
            info!("Cleaned up {} expired active tokens", removed_active);
        }

        debug!(
            "Token cleanup complete: removed {} active tokens",
            removed_active
        );
    }

    /// Get statistics about token manager state
    pub async fn get_stats(&self) -> TokenManagerStats {
        let active = self.active_tokens.read().await.len();
        let revoked = self.revoked_tokens.read().await.len();
        let revoked_jti = self.revoked_jti.read().await.len();

        TokenManagerStats {
            active_tokens: active,
            revoked_tokens: revoked,
            revoked_jti,
        }
    }

    /// Start background cleanup task
    pub fn start_cleanup_task(self: Arc<Self>) {
        let manager = Arc::clone(&self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(
                manager.cleanup_interval_secs,
            ));
            loop {
                interval.tick().await;
                manager.cleanup_expired().await;
            }
        });
    }
}

/// Token manager statistics
#[derive(Debug, Clone, Serialize)]
pub struct TokenManagerStats {
    pub active_tokens: usize,
    pub revoked_tokens: usize,
    pub revoked_jti: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_token_info(token: &str) -> TokenInfo {
        TokenInfo {
            token: token.to_string(),
            token_type: "access_token".to_string(),
            client_id: "client123".to_string(),
            subject: "user123".to_string(),
            scope: "openid profile email".to_string(),
            issued_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
            audience: Some("https://api.example.com".to_string()),
            issuer: Some("https://issuer.example.com".to_string()),
            jti: Some(format!("jti_{}", token)), // Unique JTI per token
        }
    }

    #[tokio::test]
    async fn test_token_registration() {
        let manager = TokenManager::new(3600);
        let token_info = create_test_token_info("token123");

        manager.register_token(token_info).await.unwrap();

        let stats = manager.get_stats().await;
        assert_eq!(stats.active_tokens, 1);
    }

    #[tokio::test]
    async fn test_token_introspection_active() {
        let manager = TokenManager::new(3600);
        let token_info = create_test_token_info("token123");

        manager.register_token(token_info).await.unwrap();

        let response = manager.introspect_token("token123", None).await.unwrap();
        assert!(response.active);
        assert_eq!(response.client_id.unwrap(), "client123");
    }

    #[tokio::test]
    async fn test_token_introspection_not_found() {
        let manager = TokenManager::new(3600);

        let response = manager.introspect_token("nonexistent", None).await.unwrap();
        assert!(!response.active);
    }

    #[tokio::test]
    async fn test_token_revocation() {
        let manager = TokenManager::new(3600);
        let token_info = create_test_token_info("token123");

        manager.register_token(token_info).await.unwrap();

        // Revoke the token
        manager
            .revoke_token(TokenRevocationRequest {
                token: "token123".to_string(),
                token_type_hint: Some("access_token".to_string()),
                client_id: None,
                client_secret: None,
            })
            .await
            .unwrap();

        // Check introspection returns inactive
        let response = manager.introspect_token("token123", None).await.unwrap();
        assert!(!response.active);

        // Check stats
        let stats = manager.get_stats().await;
        assert_eq!(stats.active_tokens, 0);
        assert_eq!(stats.revoked_tokens, 1);
    }

    #[tokio::test]
    async fn test_revoke_all_client_tokens() {
        let manager = TokenManager::new(3600);

        // Register multiple tokens for the same client
        for i in 0..3 {
            let mut token_info = create_test_token_info(&format!("token{}", i));
            token_info.client_id = "client123".to_string();
            manager.register_token(token_info).await.unwrap();
        }

        // Register a token for a different client
        let mut other_token = create_test_token_info("other_token");
        other_token.client_id = "client456".to_string();
        manager.register_token(other_token).await.unwrap();

        // Revoke all tokens for client123
        let count = manager.revoke_all_client_tokens("client123").await.unwrap();
        assert_eq!(count, 3);

        // Check that other client's token is still active
        let response = manager.introspect_token("other_token", None).await.unwrap();
        assert!(response.active);
    }

    #[tokio::test]
    async fn test_expired_token_introspection() {
        let manager = TokenManager::new(3600);
        let mut token_info = create_test_token_info("token123");
        token_info.expires_at = Utc::now() - chrono::Duration::hours(1); // Already expired

        manager.register_token(token_info).await.unwrap();

        let response = manager.introspect_token("token123", None).await.unwrap();
        assert!(!response.active);
    }

    #[tokio::test]
    async fn test_jti_revocation() {
        let manager = TokenManager::new(3600);
        let token_info = create_test_token_info("token123");

        manager.register_token(token_info).await.unwrap();

        // Revoke by token (which should also revoke by JTI)
        manager
            .revoke_token(TokenRevocationRequest {
                token: "token123".to_string(),
                token_type_hint: None,
                client_id: None,
                client_secret: None,
            })
            .await
            .unwrap();

        // Check JTI revocation
        assert!(manager.is_jti_revoked("jti_token123").await);
    }
}
