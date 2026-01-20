//! IDS Broker Integration
//!
//! Connects to IDS Metadata Broker for catalog federation.
//! Implements IDS-RAM 4.x Broker communication patterns.

use super::catalog::DataResource;
use super::identity::DapsClient;
use super::types::{IdsError, IdsResult, IdsUri, Party, SecurityProfile};
use chrono::{DateTime, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// IDS Message Type for broker communication
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BrokerMessageType {
    /// Connector registration/update
    ConnectorUpdateMessage,
    /// Connector going offline
    ConnectorUnavailableMessage,
    /// Resource publication
    ResourceUpdateMessage,
    /// Resource removal
    ResourceUnavailableMessage,
    /// Catalog query
    QueryMessage,
}

/// Simplified IDS Message for broker communication
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerMessage {
    /// Message ID
    #[serde(rename = "@id")]
    pub id: String,
    /// Message type
    #[serde(rename = "@type")]
    pub message_type: BrokerMessageType,
    /// Issuer connector
    pub issuer_connector: IdsUri,
    /// Timestamp
    pub issued: DateTime<Utc>,
    /// Security token
    pub security_token: Option<String>,
}

impl BrokerMessage {
    /// Create a new broker message
    pub fn new(message_type: BrokerMessageType, issuer: IdsUri) -> Self {
        Self {
            id: format!(
                "urn:ids:message:{}",
                Utc::now().timestamp_nanos_opt().unwrap_or(0)
            ),
            message_type,
            issuer_connector: issuer,
            issued: Utc::now(),
            security_token: None,
        }
    }

    /// Set security token
    pub fn with_security_token(mut self, token: impl Into<String>) -> Self {
        self.security_token = Some(token.into());
        self
    }
}

/// IDS Broker Client
///
/// Communicates with IDS Metadata Brokers for catalog federation.
pub struct BrokerClient {
    /// Broker URL
    broker_url: String,
    /// HTTP client
    http_client: Client,
    /// DAPS client for authentication
    daps_client: Arc<DapsClient>,
    /// Connector ID
    connector_id: IdsUri,
    /// Cached catalog from broker
    catalog_cache: Arc<RwLock<Option<BrokerCatalog>>>,
    /// Cache expiration
    cache_expiry: Arc<RwLock<Option<DateTime<Utc>>>>,
}

impl BrokerClient {
    /// Create a new Broker Client
    pub fn new(
        broker_url: impl Into<String>,
        connector_id: IdsUri,
        daps_client: Arc<DapsClient>,
    ) -> Self {
        Self {
            broker_url: broker_url.into(),
            http_client: Client::new(),
            daps_client,
            connector_id,
            catalog_cache: Arc::new(RwLock::new(None)),
            cache_expiry: Arc::new(RwLock::new(None)),
        }
    }

    /// Register connector with broker
    pub async fn register_connector(
        &self,
        self_description: &ConnectorSelfDescription,
    ) -> IdsResult<RegistrationResult> {
        // Get DAPS token
        let token = self.daps_client.get_token(&self.connector_id).await?;

        // Create registration message
        let message = BrokerMessage::new(
            BrokerMessageType::ConnectorUpdateMessage,
            self.connector_id.clone(),
        )
        .with_security_token(&token.access_token);

        // Send to broker
        let response = self
            .http_client
            .post(format!("{}/infrastructure", self.broker_url))
            .header("Authorization", format!("Bearer {}", token.access_token))
            .header("Content-Type", "application/json")
            .json(&BrokerRegistrationRequest {
                message,
                self_description: self_description.clone(),
            })
            .send()
            .await
            .map_err(|e| IdsError::MessageProtocolError(format!("Broker request failed: {}", e)))?;

        if response.status().is_success() {
            let result: RegistrationResult = response.json().await.map_err(|e| {
                IdsError::SerializationError(format!("Failed to parse response: {}", e))
            })?;
            Ok(result)
        } else {
            Err(IdsError::MessageProtocolError(format!(
                "Broker registration failed: {}",
                response.status()
            )))
        }
    }

    /// Unregister connector from broker
    pub async fn unregister_connector(&self) -> IdsResult<()> {
        let token = self.daps_client.get_token(&self.connector_id).await?;

        let message = BrokerMessage::new(
            BrokerMessageType::ConnectorUnavailableMessage,
            self.connector_id.clone(),
        )
        .with_security_token(&token.access_token);

        let response = self
            .http_client
            .post(format!("{}/infrastructure", self.broker_url))
            .header("Authorization", format!("Bearer {}", token.access_token))
            .header("Content-Type", "application/json")
            .json(&message)
            .send()
            .await
            .map_err(|e| IdsError::MessageProtocolError(format!("Broker request failed: {}", e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(IdsError::MessageProtocolError(format!(
                "Broker unregistration failed: {}",
                response.status()
            )))
        }
    }

    /// Publish resource to broker
    pub async fn publish_resource(&self, resource: &DataResource) -> IdsResult<()> {
        let token = self.daps_client.get_token(&self.connector_id).await?;

        let message = BrokerMessage::new(
            BrokerMessageType::ResourceUpdateMessage,
            self.connector_id.clone(),
        )
        .with_security_token(&token.access_token);

        let response = self
            .http_client
            .post(format!("{}/data", self.broker_url))
            .header("Authorization", format!("Bearer {}", token.access_token))
            .header("Content-Type", "application/json")
            .json(&BrokerResourceRequest {
                message,
                resource: resource.clone(),
            })
            .send()
            .await
            .map_err(|e| IdsError::MessageProtocolError(format!("Broker request failed: {}", e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(IdsError::MessageProtocolError(format!(
                "Resource publish failed: {}",
                response.status()
            )))
        }
    }

    /// Unpublish resource from broker
    pub async fn unpublish_resource(&self, resource_id: &IdsUri) -> IdsResult<()> {
        let token = self.daps_client.get_token(&self.connector_id).await?;

        let message = BrokerMessage::new(
            BrokerMessageType::ResourceUnavailableMessage,
            self.connector_id.clone(),
        )
        .with_security_token(&token.access_token);

        let response = self
            .http_client
            .delete(format!("{}/data/{}", self.broker_url, resource_id))
            .header("Authorization", format!("Bearer {}", token.access_token))
            .header("Content-Type", "application/json")
            .json(&message)
            .send()
            .await
            .map_err(|e| IdsError::MessageProtocolError(format!("Broker request failed: {}", e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(IdsError::MessageProtocolError(format!(
                "Resource unpublish failed: {}",
                response.status()
            )))
        }
    }

    /// Query broker catalog
    pub async fn query_catalog(&self, query: &CatalogQuery) -> IdsResult<Vec<BrokerResource>> {
        let token = self.daps_client.get_token(&self.connector_id).await?;

        let message =
            BrokerMessage::new(BrokerMessageType::QueryMessage, self.connector_id.clone())
                .with_security_token(&token.access_token);

        let response = self
            .http_client
            .post(format!("{}/data/query", self.broker_url))
            .header("Authorization", format!("Bearer {}", token.access_token))
            .header("Content-Type", "application/json")
            .json(&BrokerQueryRequest {
                message,
                query: query.clone(),
            })
            .send()
            .await
            .map_err(|e| IdsError::MessageProtocolError(format!("Broker request failed: {}", e)))?;

        if response.status().is_success() {
            let result: CatalogQueryResult = response.json().await.map_err(|e| {
                IdsError::SerializationError(format!("Failed to parse response: {}", e))
            })?;
            Ok(result.resources)
        } else {
            Err(IdsError::MessageProtocolError(format!(
                "Catalog query failed: {}",
                response.status()
            )))
        }
    }

    /// Get full broker catalog (cached)
    pub async fn get_catalog(&self, force_refresh: bool) -> IdsResult<BrokerCatalog> {
        // Check cache
        if !force_refresh {
            let cache = self.catalog_cache.read().await;
            let expiry = self.cache_expiry.read().await;

            if let (Some(catalog), Some(exp)) = (cache.as_ref(), expiry.as_ref()) {
                if *exp > Utc::now() {
                    return Ok(catalog.clone());
                }
            }
        }

        // Fetch fresh catalog
        let query = CatalogQuery::default();
        let resources = self.query_catalog(&query).await?;

        let catalog = BrokerCatalog {
            broker_url: self.broker_url.clone(),
            resources,
            fetched_at: Utc::now(),
        };

        // Update cache
        {
            let mut cache = self.catalog_cache.write().await;
            *cache = Some(catalog.clone());

            let mut expiry = self.cache_expiry.write().await;
            *expiry = Some(Utc::now() + chrono::Duration::minutes(5));
        }

        Ok(catalog)
    }

    /// Search for resources by keyword
    pub async fn search(&self, keyword: &str) -> IdsResult<Vec<BrokerResource>> {
        let query = CatalogQuery {
            keyword: Some(keyword.to_string()),
            ..Default::default()
        };
        self.query_catalog(&query).await
    }

    /// Search for resources by content type
    pub async fn search_by_type(&self, content_type: &str) -> IdsResult<Vec<BrokerResource>> {
        let query = CatalogQuery {
            content_type: Some(content_type.to_string()),
            ..Default::default()
        };
        self.query_catalog(&query).await
    }
}

/// Connector Self-Description (IDS-RAM)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorSelfDescription {
    /// Connector ID
    #[serde(rename = "@id")]
    pub id: IdsUri,

    /// Connector type
    #[serde(rename = "@type")]
    pub connector_type: String,

    /// Title
    pub title: String,

    /// Description
    pub description: String,

    /// Security profile
    pub security_profile: SecurityProfile,

    /// Curator
    pub curator: Party,

    /// Maintainer
    pub maintainer: Party,

    /// Catalog
    pub resource_catalog: Vec<DataResource>,

    /// Endpoints
    pub endpoints: Vec<ConnectorEndpoint>,

    /// Inbound model version
    pub inbound_model_version: Vec<String>,

    /// Outbound model version
    pub outbound_model_version: String,

    /// Created at
    pub created_at: DateTime<Utc>,

    /// Modified at
    pub modified_at: DateTime<Utc>,
}

impl ConnectorSelfDescription {
    /// Create a new self-description
    pub fn new(
        id: IdsUri,
        title: impl Into<String>,
        description: impl Into<String>,
        curator: Party,
    ) -> Self {
        Self {
            id,
            connector_type: "ids:BaseConnector".to_string(),
            title: title.into(),
            description: description.into(),
            security_profile: SecurityProfile::TrustSecurityProfile,
            curator: curator.clone(),
            maintainer: curator,
            resource_catalog: Vec::new(),
            endpoints: Vec::new(),
            inbound_model_version: vec!["4.2.7".to_string()],
            outbound_model_version: "4.2.7".to_string(),
            created_at: Utc::now(),
            modified_at: Utc::now(),
        }
    }

    /// Add resource to catalog
    pub fn with_resource(mut self, resource: DataResource) -> Self {
        self.resource_catalog.push(resource);
        self
    }

    /// Add endpoint
    pub fn with_endpoint(mut self, endpoint: ConnectorEndpoint) -> Self {
        self.endpoints.push(endpoint);
        self
    }

    /// Set security profile
    pub fn with_security_profile(mut self, profile: SecurityProfile) -> Self {
        self.security_profile = profile;
        self
    }
}

/// Connector Endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorEndpoint {
    /// Endpoint ID
    #[serde(rename = "@id")]
    pub id: IdsUri,

    /// Endpoint type
    #[serde(rename = "@type")]
    pub endpoint_type: String,

    /// Access URL
    pub access_url: String,

    /// Endpoint documentation
    pub endpoint_documentation: Option<String>,

    /// Path
    pub path: Option<String>,
}

impl ConnectorEndpoint {
    /// Create a new endpoint
    pub fn new(id: IdsUri, access_url: impl Into<String>) -> Self {
        Self {
            id,
            endpoint_type: "ids:ConnectorEndpoint".to_string(),
            access_url: access_url.into(),
            endpoint_documentation: None,
            path: None,
        }
    }

    /// Set endpoint documentation
    pub fn with_documentation(mut self, doc: impl Into<String>) -> Self {
        self.endpoint_documentation = Some(doc.into());
        self
    }

    /// Set path
    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// Broker Registration Request
#[derive(Debug, Serialize, Deserialize)]
struct BrokerRegistrationRequest {
    message: BrokerMessage,
    self_description: ConnectorSelfDescription,
}

/// Broker Resource Request
#[derive(Debug, Serialize, Deserialize)]
struct BrokerResourceRequest {
    message: BrokerMessage,
    resource: DataResource,
}

/// Broker Query Request
#[derive(Debug, Serialize, Deserialize)]
struct BrokerQueryRequest {
    message: BrokerMessage,
    query: CatalogQuery,
}

/// Registration Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrationResult {
    /// Registration successful
    pub success: bool,
    /// Registration ID
    pub registration_id: Option<String>,
    /// Message
    pub message: Option<String>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Catalog Query
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogQuery {
    /// Keyword search
    pub keyword: Option<String>,
    /// Content type filter
    pub content_type: Option<String>,
    /// Publisher filter
    pub publisher: Option<IdsUri>,
    /// Language filter
    pub language: Option<String>,
    /// Limit
    pub limit: Option<u32>,
    /// Offset
    pub offset: Option<u32>,
    /// SPARQL query (for advanced queries)
    pub sparql: Option<String>,
}

impl CatalogQuery {
    /// Create a new query with keyword
    pub fn keyword(keyword: impl Into<String>) -> Self {
        Self {
            keyword: Some(keyword.into()),
            ..Default::default()
        }
    }

    /// Create a SPARQL query
    pub fn sparql(query: impl Into<String>) -> Self {
        Self {
            sparql: Some(query.into()),
            ..Default::default()
        }
    }
}

/// Catalog Query Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogQueryResult {
    /// Resources found
    pub resources: Vec<BrokerResource>,
    /// Total count
    pub total: usize,
    /// Query timestamp
    pub timestamp: DateTime<Utc>,
}

/// Broker Resource (resource from broker catalog)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrokerResource {
    /// Resource ID
    #[serde(rename = "@id")]
    pub id: IdsUri,

    /// Resource title
    pub title: String,

    /// Resource description
    pub description: Option<String>,

    /// Publisher (connector ID)
    pub publisher: IdsUri,

    /// Content type
    pub content_type: Option<String>,

    /// Language
    pub language: Option<String>,

    /// Keywords
    #[serde(default)]
    pub keywords: Vec<String>,

    /// Access endpoint
    pub access_url: Option<String>,

    /// Created at
    pub created_at: DateTime<Utc>,

    /// Modified at
    pub modified_at: DateTime<Utc>,
}

/// Broker Catalog
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerCatalog {
    /// Broker URL
    pub broker_url: String,
    /// Resources in catalog
    pub resources: Vec<BrokerResource>,
    /// When catalog was fetched
    pub fetched_at: DateTime<Utc>,
}

impl BrokerCatalog {
    /// Get resources by publisher
    pub fn by_publisher(&self, publisher: &IdsUri) -> Vec<&BrokerResource> {
        self.resources
            .iter()
            .filter(|r| &r.publisher == publisher)
            .collect()
    }

    /// Get resources by keyword
    pub fn by_keyword(&self, keyword: &str) -> Vec<&BrokerResource> {
        let kw_lower = keyword.to_lowercase();
        self.resources
            .iter()
            .filter(|r| {
                r.title.to_lowercase().contains(&kw_lower)
                    || r.description
                        .as_ref()
                        .map(|d| d.to_lowercase().contains(&kw_lower))
                        .unwrap_or(false)
                    || r.keywords
                        .iter()
                        .any(|k| k.to_lowercase().contains(&kw_lower))
            })
            .collect()
    }

    /// Get resource count
    pub fn len(&self) -> usize {
        self.resources.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

/// Multi-Broker Manager
///
/// Manages connections to multiple IDS brokers for federated catalog.
pub struct MultiBrokerManager {
    /// Broker clients
    brokers: Arc<RwLock<HashMap<String, Arc<BrokerClient>>>>,
    /// Primary broker
    primary_broker: Option<String>,
}

impl MultiBrokerManager {
    /// Create a new multi-broker manager
    pub fn new() -> Self {
        Self {
            brokers: Arc::new(RwLock::new(HashMap::new())),
            primary_broker: None,
        }
    }

    /// Add a broker
    pub async fn add_broker(&self, name: impl Into<String>, client: BrokerClient) {
        let mut brokers = self.brokers.write().await;
        brokers.insert(name.into(), Arc::new(client));
    }

    /// Set primary broker
    pub fn set_primary(&mut self, name: impl Into<String>) {
        self.primary_broker = Some(name.into());
    }

    /// Get broker by name
    pub async fn get_broker(&self, name: &str) -> Option<Arc<BrokerClient>> {
        let brokers = self.brokers.read().await;
        brokers.get(name).cloned()
    }

    /// Get primary broker
    pub async fn primary(&self) -> Option<Arc<BrokerClient>> {
        if let Some(ref name) = self.primary_broker {
            self.get_broker(name).await
        } else {
            None
        }
    }

    /// Query all brokers and aggregate results
    pub async fn query_all(&self, query: &CatalogQuery) -> IdsResult<Vec<BrokerResource>> {
        let brokers = self.brokers.read().await;
        let mut all_resources = Vec::new();

        for (name, client) in brokers.iter() {
            match client.query_catalog(query).await {
                Ok(resources) => {
                    tracing::info!("Got {} resources from broker {}", resources.len(), name);
                    all_resources.extend(resources);
                }
                Err(e) => {
                    tracing::warn!("Failed to query broker {}: {}", name, e);
                }
            }
        }

        // Deduplicate by resource ID
        let mut seen = std::collections::HashSet::new();
        all_resources.retain(|r| seen.insert(r.id.as_str().to_string()));

        Ok(all_resources)
    }

    /// Register connector with all brokers
    pub async fn register_all(
        &self,
        self_description: &ConnectorSelfDescription,
    ) -> HashMap<String, IdsResult<RegistrationResult>> {
        let brokers = self.brokers.read().await;
        let mut results = HashMap::new();

        for (name, client) in brokers.iter() {
            let result = client.register_connector(self_description).await;
            results.insert(name.clone(), result);
        }

        results
    }

    /// Publish resource to all brokers
    pub async fn publish_to_all(&self, resource: &DataResource) -> HashMap<String, IdsResult<()>> {
        let brokers = self.brokers.read().await;
        let mut results = HashMap::new();

        for (name, client) in brokers.iter() {
            let result = client.publish_resource(resource).await;
            results.insert(name.clone(), result);
        }

        results
    }
}

impl Default for MultiBrokerManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_party() -> Party {
        Party {
            id: IdsUri::new("https://example.org/party/1").expect("valid uri"),
            name: "Test Party".to_string(),
            legal_name: None,
            description: None,
            contact: None,
            gaiax_participant_id: None,
        }
    }

    #[test]
    fn test_self_description_creation() {
        let id = IdsUri::new("urn:ids:connector:test").expect("valid uri");
        let sd = ConnectorSelfDescription::new(
            id,
            "Test Connector",
            "A test IDS connector",
            test_party(),
        )
        .with_security_profile(SecurityProfile::TrustPlusSecurityProfile);

        assert_eq!(sd.title, "Test Connector");
        assert_eq!(
            sd.security_profile,
            SecurityProfile::TrustPlusSecurityProfile
        );
    }

    #[test]
    fn test_endpoint_creation() {
        let id = IdsUri::new("urn:ids:endpoint:1").expect("valid uri");
        let endpoint = ConnectorEndpoint::new(id, "https://connector.example.org/api")
            .with_path("/data")
            .with_documentation("https://docs.example.org/api");

        assert_eq!(endpoint.access_url, "https://connector.example.org/api");
        assert_eq!(endpoint.path, Some("/data".to_string()));
    }

    #[test]
    fn test_catalog_query() {
        let query = CatalogQuery::keyword("sensor data").sparql.take(); // Remove sparql since we're using keyword

        let query = CatalogQuery {
            keyword: Some("sensor data".to_string()),
            limit: Some(10),
            ..Default::default()
        };

        assert_eq!(query.keyword, Some("sensor data".to_string()));
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_broker_catalog_search() {
        let catalog = BrokerCatalog {
            broker_url: "https://broker.example.org".to_string(),
            resources: vec![
                BrokerResource {
                    id: IdsUri::new("urn:ids:resource:1").expect("valid uri"),
                    title: "Temperature Sensor Data".to_string(),
                    description: Some("Real-time temperature readings".to_string()),
                    publisher: IdsUri::new("urn:ids:connector:pub1").expect("valid uri"),
                    content_type: Some("application/json".to_string()),
                    language: Some("en".to_string()),
                    keywords: vec!["temperature".to_string(), "sensor".to_string()],
                    access_url: Some("https://example.org/data/temp".to_string()),
                    created_at: Utc::now(),
                    modified_at: Utc::now(),
                },
                BrokerResource {
                    id: IdsUri::new("urn:ids:resource:2").expect("valid uri"),
                    title: "Traffic Data".to_string(),
                    description: Some("City traffic patterns".to_string()),
                    publisher: IdsUri::new("urn:ids:connector:pub2").expect("valid uri"),
                    content_type: Some("application/json".to_string()),
                    language: Some("en".to_string()),
                    keywords: vec!["traffic".to_string(), "city".to_string()],
                    access_url: Some("https://example.org/data/traffic".to_string()),
                    created_at: Utc::now(),
                    modified_at: Utc::now(),
                },
            ],
            fetched_at: Utc::now(),
        };

        let results = catalog.by_keyword("temperature");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Temperature Sensor Data");

        let results = catalog.by_keyword("sensor");
        assert_eq!(results.len(), 1);
    }
}
