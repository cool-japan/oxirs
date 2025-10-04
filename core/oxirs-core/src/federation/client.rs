//! HTTP client for executing SPARQL queries against remote endpoints

use crate::OxirsError;
use reqwest::{Client, StatusCode};
use std::time::Duration;
use tracing::{debug, info, warn};

/// Configuration for federation client
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries on failure
    pub max_retries: u32,
    /// User-Agent header
    pub user_agent: String,
    /// Accept header for SPARQL results
    pub accept: String,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            timeout_secs: 30,
            max_retries: 3,
            user_agent: format!("OxiRS/{}", env!("CARGO_PKG_VERSION")),
            accept: "application/sparql-results+json".to_string(),
        }
    }
}

/// HTTP client for federated SPARQL execution
pub struct FederationClient {
    client: Client,
    config: FederationConfig,
}

impl FederationClient {
    /// Create a new federation client with default configuration
    pub fn new() -> Result<Self, OxirsError> {
        Self::with_config(FederationConfig::default())
    }

    /// Create a new federation client with custom configuration
    pub fn with_config(config: FederationConfig) -> Result<Self, OxirsError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .user_agent(&config.user_agent)
            .build()
            .map_err(|e| OxirsError::Federation(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self { client, config })
    }

    /// Execute a SPARQL query against a remote endpoint
    ///
    /// # Arguments
    /// * `endpoint` - The SPARQL endpoint URL
    /// * `query` - The SPARQL query string
    /// * `silent` - If true, suppress errors and return empty results
    ///
    /// # Returns
    /// JSON response body as a string
    pub async fn execute_query(
        &self,
        endpoint: &str,
        query: &str,
        silent: bool,
    ) -> Result<String, OxirsError> {
        debug!("Executing federated query to endpoint: {}", endpoint);
        debug!("Query: {}", query);

        let mut last_error = None;

        for attempt in 1..=self.config.max_retries {
            match self.execute_query_once(endpoint, query).await {
                Ok(response) => {
                    info!(
                        "Successfully executed federated query (attempt {}/{})",
                        attempt, self.config.max_retries
                    );
                    return Ok(response);
                }
                Err(e) => {
                    warn!(
                        "Federated query failed (attempt {}/{}): {}",
                        attempt, self.config.max_retries, e
                    );
                    last_error = Some(e);

                    if attempt < self.config.max_retries {
                        // Exponential backoff
                        let delay = Duration::from_millis(100 * 2u64.pow(attempt - 1));
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }

        if silent {
            warn!("SERVICE SILENT: Returning empty results after all retries failed");
            // Return empty SPARQL JSON results
            Ok(r#"{"head":{"vars":[]},"results":{"bindings":[]}}"#.to_string())
        } else {
            Err(last_error.unwrap_or_else(|| {
                OxirsError::Federation("Federated query failed with unknown error".to_string())
            }))
        }
    }

    /// Execute a single query attempt
    async fn execute_query_once(&self, endpoint: &str, query: &str) -> Result<String, OxirsError> {
        // SPARQL Protocol: POST with application/x-www-form-urlencoded
        let response = self
            .client
            .post(endpoint)
            .header("Accept", &self.config.accept)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(format!("query={}", urlencoding::encode(query)))
            .send()
            .await
            .map_err(|e| {
                OxirsError::Federation(format!("Failed to send request to {}: {}", endpoint, e))
            })?;

        let status = response.status();
        let body = response
            .text()
            .await
            .map_err(|e| OxirsError::Federation(format!("Failed to read response body: {}", e)))?;

        match status {
            StatusCode::OK => Ok(body),
            StatusCode::BAD_REQUEST => Err(OxirsError::Federation(format!(
                "Bad request (400): {}",
                body
            ))),
            StatusCode::NOT_FOUND => Err(OxirsError::Federation(format!(
                "Endpoint not found (404): {}",
                endpoint
            ))),
            StatusCode::INTERNAL_SERVER_ERROR => Err(OxirsError::Federation(format!(
                "Server error (500): {}",
                body
            ))),
            StatusCode::SERVICE_UNAVAILABLE => Err(OxirsError::Federation(format!(
                "Service unavailable (503): {}",
                endpoint
            ))),
            StatusCode::GATEWAY_TIMEOUT => Err(OxirsError::Federation(format!(
                "Gateway timeout (504): {}",
                endpoint
            ))),
            _ => Err(OxirsError::Federation(format!(
                "Unexpected status code {}: {}",
                status, body
            ))),
        }
    }

    /// Check if an endpoint is reachable (health check)
    pub async fn check_endpoint(&self, endpoint: &str) -> bool {
        debug!("Checking endpoint health: {}", endpoint);

        // Simple ASK query to test connectivity
        let test_query = "ASK { ?s ?p ?o } LIMIT 1";

        match self.execute_query(endpoint, test_query, true).await {
            Ok(_) => {
                info!("Endpoint {} is healthy", endpoint);
                true
            }
            Err(e) => {
                warn!("Endpoint {} is unhealthy: {}", endpoint, e);
                false
            }
        }
    }

    /// Get endpoint capabilities via SERVICE description
    pub async fn get_service_description(&self, endpoint: &str) -> Result<String, OxirsError> {
        debug!("Fetching service description for: {}", endpoint);

        let describe_query = r#"
            PREFIX sd: <http://www.w3.org/ns/sparql-service-description#>
            DESCRIBE ?service
            WHERE {
                ?service a sd:Service
            }
        "#;

        self.execute_query(endpoint, describe_query, false).await
    }
}

impl Default for FederationClient {
    fn default() -> Self {
        Self::new().expect("Failed to create default federation client")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = FederationClient::new();
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_client_with_custom_config() {
        let config = FederationConfig {
            timeout_secs: 60,
            max_retries: 5,
            user_agent: "TestAgent/1.0".to_string(),
            accept: "application/sparql-results+json".to_string(),
        };

        let client = FederationClient::with_config(config);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_silent_mode_on_error() {
        let client = FederationClient::new().unwrap();

        // Non-existent endpoint should return empty results in silent mode
        let result = client
            .execute_query(
                "http://nonexistent.example.org/sparql",
                "SELECT * WHERE { ?s ?p ?o }",
                true,
            )
            .await;

        assert!(result.is_ok());
        let body = result.unwrap();
        assert!(body.contains("bindings"));
    }
}
