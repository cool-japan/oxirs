//! did:web Method implementation
//!
//! did:web resolves DIDs by fetching a DID Document from a well-known
//! HTTPS URL derived from the domain in the DID.

use super::DidMethod;
use crate::did::{Did, DidDocument};
use crate::{DidError, DidResult};
use async_trait::async_trait;

/// did:web method resolver
#[cfg(feature = "did-web")]
pub struct DidWebMethod {
    /// HTTP client
    client: reqwest::Client,
    /// Timeout in seconds
    timeout_secs: u64,
}

#[cfg(feature = "did-web")]
impl Default for DidWebMethod {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "did-web")]
impl DidWebMethod {
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            timeout_secs: 30,
        }
    }

    pub fn with_timeout(mut self, timeout_secs: u64) -> Self {
        self.timeout_secs = timeout_secs;
        self
    }

    /// Convert did:web to URL for DID Document
    ///
    /// According to the did:web spec:
    /// - Colons (:) in the domain:port are percent-encoded as %3A
    /// - Colons (:) between path segments are NOT encoded
    /// - Slashes (/) in path segments are percent-encoded as %2F
    pub fn did_to_url(&self, did: &Did) -> DidResult<String> {
        let method_specific_id = did.method_specific_id();

        if method_specific_id.is_empty() {
            return Err(DidError::InvalidFormat(
                "Empty method-specific-id".to_string(),
            ));
        }

        // Split by unencoded colons first to get path segments
        // The first segment is the domain (which may contain %3A for port)
        let parts: Vec<&str> = method_specific_id.split(':').collect();

        // Decode the domain part (may contain %3A for port)
        let domain = parts[0].replace("%3A", ":").replace("%2F", "/");

        // Decode and join path segments
        let path = if parts.len() > 1 {
            parts[1..]
                .iter()
                .map(|p| p.replace("%2F", "/"))
                .collect::<Vec<_>>()
                .join("/")
        } else {
            ".well-known".to_string()
        };

        // Construct URL
        let url = if path == ".well-known" {
            format!("https://{}/.well-known/did.json", domain)
        } else {
            format!("https://{}/{}/did.json", domain, path)
        };

        Ok(url)
    }
}

#[cfg(feature = "did-web")]
#[async_trait]
impl DidMethod for DidWebMethod {
    fn method_name(&self) -> &str {
        "web"
    }

    async fn resolve(&self, did: &Did) -> DidResult<DidDocument> {
        if !self.supports(did) {
            return Err(DidError::UnsupportedMethod(did.method().to_string()));
        }

        let url = self.did_to_url(did)?;

        // Fetch DID Document
        let response = self
            .client
            .get(&url)
            .timeout(std::time::Duration::from_secs(self.timeout_secs))
            .header("Accept", "application/did+json, application/json")
            .send()
            .await
            .map_err(|e| DidError::NetworkError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(DidError::ResolutionFailed(format!(
                "HTTP {} from {}",
                response.status(),
                url
            )));
        }

        let body = response
            .text()
            .await
            .map_err(|e| DidError::NetworkError(e.to_string()))?;

        DidDocument::from_json(&body)
    }
}

#[cfg(all(test, feature = "did-web"))]
mod tests {
    use super::*;

    #[test]
    fn test_did_to_url_simple() {
        let method = DidWebMethod::new();

        let did = Did::new("did:web:example.com").unwrap();
        let url = method.did_to_url(&did).unwrap();

        assert_eq!(url, "https://example.com/.well-known/did.json");
    }

    #[test]
    fn test_did_to_url_with_path() {
        let method = DidWebMethod::new();

        let did = Did::new("did:web:example.com:users:alice").unwrap();
        let url = method.did_to_url(&did).unwrap();

        assert_eq!(url, "https://example.com/users/alice/did.json");
    }

    #[test]
    fn test_did_to_url_with_port() {
        let method = DidWebMethod::new();

        let did = Did::new("did:web:example.com%3A8080").unwrap();
        let url = method.did_to_url(&did).unwrap();

        assert_eq!(url, "https://example.com:8080/.well-known/did.json");
    }
}
