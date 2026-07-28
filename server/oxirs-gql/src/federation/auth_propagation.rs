// Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Cross-Service Authentication Propagation for Federation
//!
//! This module provides authentication credential propagation from the gateway
//! to federated subgraphs, ensuring secure and consistent authentication across
//! the distributed GraphQL architecture.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Authentication scheme types supported in federation
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthScheme {
    /// Bearer token (JWT or OAuth2)
    Bearer,
    /// API key in header
    ApiKey,
    /// Basic authentication
    Basic,
    /// Custom authentication scheme
    Custom(String),
}

/// Authentication credential to propagate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthCredential {
    /// Authentication scheme
    pub scheme: AuthScheme,
    /// Credential value (token, key, etc.)
    pub value: String,
    /// Optional additional headers
    pub headers: HashMap<String, String>,
    /// Optional metadata
    pub metadata: HashMap<String, String>,
}

impl AuthCredential {
    /// Create a new Bearer token credential
    pub fn bearer(token: String) -> Self {
        Self {
            scheme: AuthScheme::Bearer,
            value: token,
            headers: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new API key credential
    pub fn api_key(key: String) -> Self {
        Self {
            scheme: AuthScheme::ApiKey,
            value: key,
            headers: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create a new Basic auth credential
    pub fn basic(username: String, password: String) -> Self {
        use base64::Engine;
        let credentials =
            base64::engine::general_purpose::STANDARD.encode(format!("{}:{}", username, password));
        Self {
            scheme: AuthScheme::Basic,
            value: credentials,
            headers: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a custom header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Format as HTTP Authorization header value
    pub fn to_authorization_header(&self) -> String {
        match &self.scheme {
            AuthScheme::Bearer => format!("Bearer {}", self.value),
            AuthScheme::ApiKey => self.value.clone(),
            AuthScheme::Basic => format!("Basic {}", self.value),
            AuthScheme::Custom(scheme) => format!("{} {}", scheme, self.value),
        }
    }
}

/// Authentication propagation strategy
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationStrategy {
    /// Forward credentials unchanged
    Forward,
    /// Transform credentials per service
    Transform,
    /// Exchange for service-specific tokens
    Exchange,
    /// Selective propagation based on service
    Selective,
}

/// Service-specific authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAuthConfig {
    /// Service name
    pub service_name: String,
    /// Whether to propagate auth to this service
    pub enabled: bool,
    /// Propagation strategy
    pub strategy: PropagationStrategy,
    /// Header name for authentication (default: "Authorization")
    pub header_name: String,
    /// Optional token transformation function name
    pub transform_fn: Option<String>,
    /// Additional required headers
    pub required_headers: Vec<String>,
}

impl Default for ServiceAuthConfig {
    fn default() -> Self {
        Self {
            service_name: String::new(),
            enabled: true,
            strategy: PropagationStrategy::Forward,
            header_name: "Authorization".to_string(),
            transform_fn: None,
            required_headers: Vec::new(),
        }
    }
}

/// Token transformation result
#[derive(Debug, Clone)]
pub struct TransformedCredential {
    /// Original credential
    pub original: AuthCredential,
    /// Transformed credential
    pub transformed: AuthCredential,
    /// Transformation metadata
    pub metadata: HashMap<String, String>,
}

/// Authentication propagation manager
pub struct AuthPropagationManager {
    /// Service configurations
    service_configs: Arc<RwLock<HashMap<String, ServiceAuthConfig>>>,
    /// Token cache for exchanged tokens
    token_cache: Arc<RwLock<HashMap<String, AuthCredential>>>,
    /// Token transformation registry
    transformers: Arc<RwLock<HashMap<String, Box<dyn TokenTransformer + Send + Sync>>>>,
}

/// Token transformer trait for service-specific transformations
pub trait TokenTransformer {
    /// Transform a credential for a specific service
    fn transform(&self, credential: &AuthCredential, service_name: &str) -> Result<AuthCredential>;
}

impl AuthPropagationManager {
    /// Create a new authentication propagation manager
    pub fn new() -> Self {
        Self {
            service_configs: Arc::new(RwLock::new(HashMap::new())),
            token_cache: Arc::new(RwLock::new(HashMap::new())),
            transformers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a service authentication configuration
    pub async fn register_service(&self, config: ServiceAuthConfig) -> Result<()> {
        let mut configs = self.service_configs.write().await;
        configs.insert(config.service_name.clone(), config);
        Ok(())
    }

    /// Register a token transformer
    pub async fn register_transformer(
        &self,
        name: String,
        transformer: Box<dyn TokenTransformer + Send + Sync>,
    ) -> Result<()> {
        let mut transformers = self.transformers.write().await;
        transformers.insert(name, transformer);
        Ok(())
    }

    /// Propagate authentication to a service
    pub async fn propagate(
        &self,
        credential: &AuthCredential,
        service_name: &str,
    ) -> Result<Option<HashMap<String, String>>> {
        let configs = self.service_configs.read().await;
        let config = configs
            .get(service_name)
            .ok_or_else(|| anyhow!("Service '{}' not registered", service_name))?;

        if !config.enabled {
            return Ok(None);
        }

        let propagated_credential = match config.strategy {
            PropagationStrategy::Forward => credential.clone(),
            PropagationStrategy::Transform => self.transform_credential(credential, config).await?,
            PropagationStrategy::Exchange => {
                self.exchange_credential(credential, service_name).await?
            }
            PropagationStrategy::Selective => {
                if self.should_propagate(credential, config).await {
                    credential.clone()
                } else {
                    return Ok(None);
                }
            }
        };

        let mut headers = HashMap::new();
        headers.insert(
            config.header_name.clone(),
            propagated_credential.to_authorization_header(),
        );

        // Add custom headers from credential
        for (key, value) in &propagated_credential.headers {
            headers.insert(key.clone(), value.clone());
        }

        Ok(Some(headers))
    }

    /// Transform credential based on service configuration
    async fn transform_credential(
        &self,
        credential: &AuthCredential,
        config: &ServiceAuthConfig,
    ) -> Result<AuthCredential> {
        if let Some(transform_fn) = &config.transform_fn {
            let transformers = self.transformers.read().await;
            let transformer = transformers
                .get(transform_fn)
                .ok_or_else(|| anyhow!("Transformer '{}' not found", transform_fn))?;
            transformer.transform(credential, &config.service_name)
        } else {
            Ok(credential.clone())
        }
    }

    /// Exchange credential for service-specific token
    async fn exchange_credential(
        &self,
        credential: &AuthCredential,
        service_name: &str,
    ) -> Result<AuthCredential> {
        // Check cache first
        let cache_key = format!("{}:{}", service_name, credential.value);
        {
            let cache = self.token_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }

        // Perform token exchange (placeholder for actual implementation)
        // In a real implementation, this would call a token exchange service
        let exchanged = credential.clone();

        // Cache the exchanged token
        let mut cache = self.token_cache.write().await;
        cache.insert(cache_key, exchanged.clone());

        Ok(exchanged)
    }

    /// Determine if credential should be propagated
    async fn should_propagate(
        &self,
        credential: &AuthCredential,
        config: &ServiceAuthConfig,
    ) -> bool {
        // Check if credential has required metadata/headers
        for required_header in &config.required_headers {
            if !credential.headers.contains_key(required_header) {
                return false;
            }
        }
        true
    }

    /// Clear token cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.token_cache.write().await;
        cache.clear();
        Ok(())
    }

    /// Get service configuration
    pub async fn get_service_config(&self, service_name: &str) -> Option<ServiceAuthConfig> {
        let configs = self.service_configs.read().await;
        configs.get(service_name).cloned()
    }

    /// List all registered services
    pub async fn list_services(&self) -> Vec<String> {
        let configs = self.service_configs.read().await;
        configs.keys().cloned().collect()
    }

    /// Remove a service configuration
    pub async fn unregister_service(&self, service_name: &str) -> Result<()> {
        let mut configs = self.service_configs.write().await;
        configs.remove(service_name);
        Ok(())
    }
}

impl Default for AuthPropagationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple JWT token transformer.
///
/// Applies an issuer (`iss`) claim remapping and re-signs the token with a
/// configured HMAC-SHA256 signing key. If an issuer mapping is configured but
/// no signing key is available, [`TokenTransformer::transform`] fails loud
/// rather than silently forwarding the token with its mapping ignored (which
/// would make the configuration a lie).
pub struct JwtTransformer {
    /// Issuer transformation map
    issuer_map: HashMap<String, String>,
    /// HMAC-SHA256 signing key used to re-sign a remapped token.
    signing_key: Option<Vec<u8>>,
}

impl JwtTransformer {
    /// Create a new JWT transformer
    pub fn new() -> Self {
        Self {
            issuer_map: HashMap::new(),
            signing_key: None,
        }
    }

    /// Add issuer mapping
    pub fn with_issuer_mapping(mut self, from: String, to: String) -> Self {
        self.issuer_map.insert(from, to);
        self
    }

    /// Set the HMAC-SHA256 signing key used to re-sign remapped tokens.
    pub fn with_signing_key(mut self, key: impl Into<Vec<u8>>) -> Self {
        self.signing_key = Some(key.into());
        self
    }

    /// base64url (no padding) encode.
    fn b64url_encode(data: &[u8]) -> String {
        use base64::Engine;
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(data)
    }

    /// base64url (no padding) decode.
    fn b64url_decode(data: &str) -> Result<Vec<u8>> {
        use base64::Engine;
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(data)
            .map_err(|e| anyhow::anyhow!("Invalid base64url segment: {e}"))
    }

    /// HMAC-SHA256 sign.
    fn sign(key: &[u8], signing_input: &str) -> Result<Vec<u8>> {
        use hmac::{Hmac, KeyInit, Mac};
        use sha2::Sha256;
        let mut mac = <Hmac<Sha256>>::new_from_slice(key)
            .map_err(|e| anyhow::anyhow!("Invalid HMAC key: {e}"))?;
        mac.update(signing_input.as_bytes());
        Ok(mac.finalize().into_bytes().to_vec())
    }

    /// Decode the payload, remap the `iss` claim per `issuer_map`, and re-sign
    /// the token with `signing_key`. Returns the new compact JWT string.
    fn remap_and_resign(&self, token: &str) -> Result<String> {
        let signing_key = self.signing_key.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "JwtTransformer has an issuer mapping configured but no signing key; \
                 refusing to forward a token whose configured issuer remap cannot be applied"
            )
        })?;

        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(anyhow::anyhow!(
                "Malformed JWT: expected 3 segments, got {}",
                parts.len()
            ));
        }

        // Decode and remap the payload's issuer claim.
        let payload_bytes = Self::b64url_decode(parts[1])?;
        let mut payload: serde_json::Value = serde_json::from_slice(&payload_bytes)
            .map_err(|e| anyhow::anyhow!("Invalid JWT payload JSON: {e}"))?;

        if let Some(iss) = payload.get("iss").and_then(|v| v.as_str()) {
            if let Some(mapped) = self.issuer_map.get(iss) {
                payload["iss"] = serde_json::Value::String(mapped.clone());
            }
        }

        // Re-encode the payload and re-sign (header segment is preserved).
        let new_payload_b64 = Self::b64url_encode(serde_json::to_string(&payload)?.as_bytes());
        let signing_input = format!("{}.{}", parts[0], new_payload_b64);
        let signature = Self::sign(signing_key, &signing_input)?;
        let signature_b64 = Self::b64url_encode(&signature);

        Ok(format!("{signing_input}.{signature_b64}"))
    }
}

impl Default for JwtTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenTransformer for JwtTransformer {
    fn transform(
        &self,
        credential: &AuthCredential,
        _service_name: &str,
    ) -> Result<AuthCredential> {
        // No mapping configured => nothing to do; forward unchanged.
        if self.issuer_map.is_empty() {
            return Ok(credential.clone());
        }

        // A mapping is configured but this credential is not a JWT bearer token,
        // so the mapping cannot be applied — fail loud rather than silently
        // forwarding an untransformed credential.
        if credential.scheme != AuthScheme::Bearer {
            return Err(anyhow::anyhow!(
                "JwtTransformer issuer mapping is configured but credential scheme is {:?}, \
                 not Bearer; cannot apply mapping",
                credential.scheme
            ));
        }

        let new_token = self.remap_and_resign(&credential.value)?;
        let mut transformed = credential.clone();
        transformed.value = new_token;
        Ok(transformed)
    }
}

/// API key transformer
pub struct ApiKeyTransformer {
    /// Service-specific key mapping
    key_map: HashMap<String, String>,
}

impl ApiKeyTransformer {
    /// Create a new API key transformer
    pub fn new() -> Self {
        Self {
            key_map: HashMap::new(),
        }
    }

    /// Add key mapping for a service
    pub fn with_key_mapping(mut self, service: String, api_key: String) -> Self {
        self.key_map.insert(service, api_key);
        self
    }
}

impl Default for ApiKeyTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenTransformer for ApiKeyTransformer {
    fn transform(
        &self,
        _credential: &AuthCredential,
        service_name: &str,
    ) -> Result<AuthCredential> {
        // Replace with service-specific API key
        let api_key = self
            .key_map
            .get(service_name)
            .ok_or_else(|| anyhow!("No API key mapping for service '{}'", service_name))?;
        Ok(AuthCredential::api_key(api_key.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auth_credential_bearer() {
        let cred = AuthCredential::bearer("test_token".to_string());
        assert_eq!(cred.scheme, AuthScheme::Bearer);
        assert_eq!(cred.to_authorization_header(), "Bearer test_token");
    }

    #[test]
    fn test_auth_credential_api_key() {
        let cred = AuthCredential::api_key("my_api_key".to_string());
        assert_eq!(cred.scheme, AuthScheme::ApiKey);
        assert_eq!(cred.to_authorization_header(), "my_api_key");
    }

    #[test]
    fn test_auth_credential_basic() {
        let cred = AuthCredential::basic("user".to_string(), "pass".to_string());
        assert_eq!(cred.scheme, AuthScheme::Basic);
        assert!(cred.to_authorization_header().starts_with("Basic "));
    }

    #[test]
    fn test_auth_credential_with_headers() {
        let cred = AuthCredential::bearer("token".to_string())
            .with_header("X-Custom".to_string(), "value".to_string());
        assert_eq!(cred.headers.get("X-Custom"), Some(&"value".to_string()));
    }

    #[tokio::test]
    async fn test_auth_propagation_manager_register_service() {
        let manager = AuthPropagationManager::new();
        let config = ServiceAuthConfig {
            service_name: "test_service".to_string(),
            enabled: true,
            strategy: PropagationStrategy::Forward,
            header_name: "Authorization".to_string(),
            transform_fn: None,
            required_headers: vec![],
        };

        manager
            .register_service(config)
            .await
            .expect("should succeed");
        let retrieved = manager.get_service_config("test_service").await;
        assert!(retrieved.is_some());
        assert_eq!(
            retrieved.expect("should succeed").service_name,
            "test_service"
        );
    }

    #[tokio::test]
    async fn test_propagate_forward_strategy() {
        let manager = AuthPropagationManager::new();
        let config = ServiceAuthConfig {
            service_name: "test_service".to_string(),
            enabled: true,
            strategy: PropagationStrategy::Forward,
            header_name: "Authorization".to_string(),
            transform_fn: None,
            required_headers: vec![],
        };
        manager
            .register_service(config)
            .await
            .expect("should succeed");

        let cred = AuthCredential::bearer("test_token".to_string());
        let headers = manager
            .propagate(&cred, "test_service")
            .await
            .expect("should succeed");
        assert!(headers.is_some());
        let headers = headers.expect("should succeed");
        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer test_token".to_string())
        );
    }

    #[tokio::test]
    async fn test_propagate_disabled_service() {
        let manager = AuthPropagationManager::new();
        let config = ServiceAuthConfig {
            service_name: "test_service".to_string(),
            enabled: false,
            strategy: PropagationStrategy::Forward,
            header_name: "Authorization".to_string(),
            transform_fn: None,
            required_headers: vec![],
        };
        manager
            .register_service(config)
            .await
            .expect("should succeed");

        let cred = AuthCredential::bearer("test_token".to_string());
        let headers = manager
            .propagate(&cred, "test_service")
            .await
            .expect("should succeed");
        assert!(headers.is_none());
    }

    #[tokio::test]
    async fn test_list_services() {
        let manager = AuthPropagationManager::new();
        let config1 = ServiceAuthConfig {
            service_name: "service1".to_string(),
            ..Default::default()
        };
        let config2 = ServiceAuthConfig {
            service_name: "service2".to_string(),
            ..Default::default()
        };

        manager
            .register_service(config1)
            .await
            .expect("should succeed");
        manager
            .register_service(config2)
            .await
            .expect("should succeed");

        let services = manager.list_services().await;
        assert_eq!(services.len(), 2);
        assert!(services.contains(&"service1".to_string()));
        assert!(services.contains(&"service2".to_string()));
    }

    #[tokio::test]
    async fn test_unregister_service() {
        let manager = AuthPropagationManager::new();
        let config = ServiceAuthConfig {
            service_name: "test_service".to_string(),
            ..Default::default()
        };
        manager
            .register_service(config)
            .await
            .expect("should succeed");

        assert!(manager.get_service_config("test_service").await.is_some());
        manager
            .unregister_service("test_service")
            .await
            .expect("should succeed");
        assert!(manager.get_service_config("test_service").await.is_none());
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let manager = AuthPropagationManager::new();
        manager.clear_cache().await.expect("should succeed");
        // Just verify it doesn't panic
    }

    fn make_jwt(iss: &str) -> String {
        use base64::Engine;
        let enc = |s: &str| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(s.as_bytes());
        let header = enc(r#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = enc(&format!(r#"{{"iss":"{iss}","sub":"user"}}"#));
        // Signature is irrelevant to the transformer (it re-signs); use a stub.
        format!("{header}.{payload}.stubsig")
    }

    fn decode_iss(token: &str) -> String {
        use base64::Engine;
        let payload = token.split('.').nth(1).expect("payload segment");
        let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(payload)
            .expect("decode payload");
        let json: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
        json["iss"].as_str().expect("iss").to_string()
    }

    #[tokio::test]
    async fn test_jwt_transformer_no_mapping_is_passthrough() {
        let transformer = JwtTransformer::new();
        let cred = AuthCredential::bearer("any_token".to_string());
        let result = transformer
            .transform(&cred, "test_service")
            .expect("passthrough ok");
        assert_eq!(result.value, "any_token");
    }

    #[tokio::test]
    async fn regression_jwt_issuer_mapping_without_key_fails_loud() {
        let transformer =
            JwtTransformer::new().with_issuer_mapping("issuer1".to_string(), "issuer2".to_string());
        let cred = AuthCredential::bearer(make_jwt("issuer1"));
        // Mapping configured but no signing key => must error, never silently ignore.
        assert!(transformer.transform(&cred, "svc").is_err());
    }

    #[tokio::test]
    async fn regression_jwt_issuer_mapping_applied_and_resigned() {
        let transformer = JwtTransformer::new()
            .with_issuer_mapping("issuer1".to_string(), "issuer2".to_string())
            .with_signing_key(b"service-signing-secret".to_vec());
        let cred = AuthCredential::bearer(make_jwt("issuer1"));
        let result = transformer.transform(&cred, "svc").expect("transform ok");

        // The issuer claim must actually be rewritten.
        assert_eq!(decode_iss(&result.value), "issuer2");
        // Token must be re-signed (3 segments, new signature).
        let parts: Vec<&str> = result.value.split('.').collect();
        assert_eq!(parts.len(), 3);
        assert_ne!(parts[2], "stubsig", "token must be re-signed");
    }

    #[tokio::test]
    async fn regression_jwt_unmapped_issuer_preserved() {
        let transformer = JwtTransformer::new()
            .with_issuer_mapping("issuerX".to_string(), "issuerY".to_string())
            .with_signing_key(b"k".to_vec());
        let cred = AuthCredential::bearer(make_jwt("issuer1"));
        let result = transformer.transform(&cred, "svc").expect("ok");
        // issuer1 is not in the map, so it stays as-is.
        assert_eq!(decode_iss(&result.value), "issuer1");
    }

    #[tokio::test]
    async fn test_api_key_transformer() {
        let transformer = ApiKeyTransformer::new()
            .with_key_mapping("test_service".to_string(), "service_api_key".to_string());
        let cred = AuthCredential::api_key("original_key".to_string());
        let result = transformer
            .transform(&cred, "test_service")
            .expect("should succeed");
        assert_eq!(result.value, "service_api_key");
    }

    #[tokio::test]
    async fn test_propagate_with_custom_headers() {
        let manager = AuthPropagationManager::new();
        let config = ServiceAuthConfig {
            service_name: "test_service".to_string(),
            enabled: true,
            strategy: PropagationStrategy::Forward,
            header_name: "Authorization".to_string(),
            transform_fn: None,
            required_headers: vec![],
        };
        manager
            .register_service(config)
            .await
            .expect("should succeed");

        let cred = AuthCredential::bearer("token".to_string())
            .with_header("X-Custom-Header".to_string(), "custom_value".to_string());

        let headers = manager
            .propagate(&cred, "test_service")
            .await
            .expect("should succeed");
        assert!(headers.is_some());
        let headers = headers.expect("should succeed");
        assert_eq!(
            headers.get("X-Custom-Header"),
            Some(&"custom_value".to_string())
        );
    }

    #[tokio::test]
    async fn test_selective_propagation() {
        let manager = AuthPropagationManager::new();
        let config = ServiceAuthConfig {
            service_name: "test_service".to_string(),
            enabled: true,
            strategy: PropagationStrategy::Selective,
            header_name: "Authorization".to_string(),
            transform_fn: None,
            required_headers: vec!["X-Required".to_string()],
        };
        manager
            .register_service(config)
            .await
            .expect("should succeed");

        // Should not propagate without required header
        let cred1 = AuthCredential::bearer("token".to_string());
        let headers1 = manager
            .propagate(&cred1, "test_service")
            .await
            .expect("should succeed");
        assert!(headers1.is_none());

        // Should propagate with required header
        let cred2 = AuthCredential::bearer("token".to_string())
            .with_header("X-Required".to_string(), "value".to_string());
        let headers2 = manager
            .propagate(&cred2, "test_service")
            .await
            .expect("should succeed");
        assert!(headers2.is_some());
    }

    #[tokio::test]
    async fn test_register_transformer() {
        let manager = AuthPropagationManager::new();
        let transformer = Box::new(JwtTransformer::new());
        manager
            .register_transformer("jwt".to_string(), transformer)
            .await
            .expect("should succeed");
        // Verify it doesn't panic
    }

    #[tokio::test]
    async fn test_exchange_strategy() {
        let manager = AuthPropagationManager::new();
        let config = ServiceAuthConfig {
            service_name: "test_service".to_string(),
            enabled: true,
            strategy: PropagationStrategy::Exchange,
            header_name: "Authorization".to_string(),
            transform_fn: None,
            required_headers: vec![],
        };
        manager
            .register_service(config)
            .await
            .expect("should succeed");

        let cred = AuthCredential::bearer("original_token".to_string());
        let headers = manager
            .propagate(&cred, "test_service")
            .await
            .expect("should succeed");
        assert!(headers.is_some());
    }
}
