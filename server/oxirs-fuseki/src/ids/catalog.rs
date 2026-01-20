//! IDS Resource Catalog
//!
//! DCAT-AP (Data Catalog Vocabulary - Application Profile) implementation
//! for IDS data space catalogs.

use super::types::{IdsResult, IdsUri};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Resource Catalog (DCAT-AP)
pub struct ResourceCatalog {
    resources: Arc<RwLock<HashMap<String, DataResource>>>,
}

impl Default for ResourceCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceCatalog {
    /// Create a new resource catalog
    pub fn new() -> Self {
        Self {
            resources: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a resource to the catalog
    pub async fn add_resource(&self, resource: DataResource) {
        self.resources
            .write()
            .await
            .insert(resource.id.as_str().to_string(), resource);
    }

    /// Get resource by string ID
    pub async fn get_resource(&self, id: &str) -> Option<DataResource> {
        self.resources.read().await.get(id).cloned()
    }

    /// Get resource by IdsUri
    pub async fn get_resource_by_uri(&self, id: &IdsUri) -> Option<DataResource> {
        self.resources.read().await.get(id.as_str()).cloned()
    }

    /// List all resources
    pub async fn list_resources(&self) -> Vec<DataResource> {
        self.resources.read().await.values().cloned().collect()
    }

    /// List all resources (alias for list_resources)
    pub async fn list_all(&self) -> Vec<DataResource> {
        self.list_resources().await
    }

    /// Remove a resource by string ID
    pub async fn remove_resource(&self, id: &str) -> bool {
        self.resources.write().await.remove(id).is_some()
    }

    /// Remove a resource by IdsUri
    pub async fn remove_resource_by_uri(&self, id: &IdsUri) -> bool {
        self.remove_resource(id.as_str()).await
    }

    /// Update a resource
    pub async fn update_resource(&self, resource: DataResource) -> bool {
        let mut resources = self.resources.write().await;
        let id = resource.id.as_str().to_string();
        if let std::collections::hash_map::Entry::Occupied(mut e) = resources.entry(id) {
            e.insert(resource);
            true
        } else {
            false
        }
    }

    /// Get resource count
    pub async fn count(&self) -> usize {
        self.resources.read().await.len()
    }

    /// Check if resource exists
    pub async fn contains(&self, id: &str) -> bool {
        self.resources.read().await.contains_key(id)
    }

    /// Search resources by keyword
    pub async fn search(&self, keyword: &str) -> Vec<DataResource> {
        let kw_lower = keyword.to_lowercase();
        self.resources
            .read()
            .await
            .values()
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
            .cloned()
            .collect()
    }

    /// Filter resources by content type
    pub async fn filter_by_content_type(&self, content_type: &str) -> Vec<DataResource> {
        self.resources
            .read()
            .await
            .values()
            .filter(|r| {
                r.content_type
                    .as_ref()
                    .map(|ct| ct == content_type)
                    .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Filter resources by language
    pub async fn filter_by_language(&self, language: &str) -> Vec<DataResource> {
        self.resources
            .read()
            .await
            .values()
            .filter(|r| r.language.as_ref().map(|l| l == language).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// Clear all resources
    pub async fn clear(&self) {
        self.resources.write().await.clear();
    }
}

/// Data Resource (DCAT-AP dcat:Resource)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DataResource {
    /// Resource identifier
    #[serde(rename = "@id")]
    pub id: IdsUri,

    /// Resource title
    pub title: String,

    /// Resource description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Keywords/tags
    #[serde(default)]
    pub keywords: Vec<String>,

    /// Publisher (connector ID)
    pub publisher: IdsUri,

    /// Available distributions
    #[serde(default)]
    pub distributions: Vec<Distribution>,

    /// Content type (MIME type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,

    /// Language (ISO 639-1)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,

    /// Access URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub access_url: Option<String>,

    /// Download URL
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,

    /// File size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_size: Option<u64>,

    /// Checksum (SHA-256)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,

    /// License
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Version
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last modification timestamp
    pub modified_at: DateTime<Utc>,
}

impl DataResource {
    /// Create a new data resource
    pub fn new(id: IdsUri, title: impl Into<String>, publisher: IdsUri) -> Self {
        let now = Utc::now();
        Self {
            id,
            title: title.into(),
            description: None,
            keywords: Vec::new(),
            publisher,
            distributions: Vec::new(),
            content_type: None,
            language: None,
            access_url: None,
            download_url: None,
            byte_size: None,
            checksum: None,
            license: None,
            version: None,
            created_at: now,
            modified_at: now,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Add keyword
    pub fn with_keyword(mut self, keyword: impl Into<String>) -> Self {
        self.keywords.push(keyword.into());
        self
    }

    /// Set keywords
    pub fn with_keywords(mut self, keywords: Vec<String>) -> Self {
        self.keywords = keywords;
        self
    }

    /// Set content type
    pub fn with_content_type(mut self, content_type: impl Into<String>) -> Self {
        self.content_type = Some(content_type.into());
        self
    }

    /// Set language
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set access URL
    pub fn with_access_url(mut self, url: impl Into<String>) -> Self {
        self.access_url = Some(url.into());
        self
    }

    /// Set download URL
    pub fn with_download_url(mut self, url: impl Into<String>) -> Self {
        self.download_url = Some(url.into());
        self
    }

    /// Set byte size
    pub fn with_byte_size(mut self, size: u64) -> Self {
        self.byte_size = Some(size);
        self
    }

    /// Set checksum
    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }

    /// Set license
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Set version
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = Some(version.into());
        self
    }

    /// Add distribution
    pub fn with_distribution(mut self, distribution: Distribution) -> Self {
        self.distributions.push(distribution);
        self
    }

    /// Update modified timestamp
    pub fn touch(&mut self) {
        self.modified_at = Utc::now();
    }

    /// Alias for created_at (for API compatibility)
    pub fn created(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Alias for modified_at (for API compatibility)
    pub fn modified(&self) -> DateTime<Utc> {
        self.modified_at
    }
}

/// Distribution (access method for a resource)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Distribution {
    /// Distribution identifier
    #[serde(rename = "@id", skip_serializing_if = "Option::is_none")]
    pub id: Option<IdsUri>,

    /// Access URL
    pub access_url: String,

    /// Format (e.g., "JSON", "CSV", "RDF")
    pub format: String,

    /// Media type (MIME type)
    pub media_type: String,

    /// Download URL (if different from access URL)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub download_url: Option<String>,

    /// File size in bytes
    #[serde(skip_serializing_if = "Option::is_none")]
    pub byte_size: Option<u64>,

    /// Checksum
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}

impl Distribution {
    /// Create a new distribution
    pub fn new(
        access_url: impl Into<String>,
        format: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        Self {
            id: None,
            access_url: access_url.into(),
            format: format.into(),
            media_type: media_type.into(),
            download_url: None,
            byte_size: None,
            checksum: None,
        }
    }

    /// Set ID
    pub fn with_id(mut self, id: IdsUri) -> Self {
        self.id = Some(id);
        self
    }

    /// Set download URL
    pub fn with_download_url(mut self, url: impl Into<String>) -> Self {
        self.download_url = Some(url.into());
        self
    }

    /// Set byte size
    pub fn with_byte_size(mut self, size: u64) -> Self {
        self.byte_size = Some(size);
        self
    }

    /// Set checksum
    pub fn with_checksum(mut self, checksum: impl Into<String>) -> Self {
        self.checksum = Some(checksum.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_uri() -> IdsUri {
        IdsUri::new("urn:ids:resource:test").expect("valid uri")
    }

    fn test_publisher() -> IdsUri {
        IdsUri::new("urn:ids:connector:pub").expect("valid uri")
    }

    #[tokio::test]
    async fn test_catalog_crud() {
        let catalog = ResourceCatalog::new();

        let resource = DataResource::new(test_uri(), "Test Resource", test_publisher())
            .with_description("A test resource")
            .with_keyword("test");

        // Create
        catalog.add_resource(resource.clone()).await;
        assert_eq!(catalog.count().await, 1);

        // Read
        let found = catalog.get_resource("urn:ids:resource:test").await;
        assert!(found.is_some());
        assert_eq!(found.expect("should exist").title, "Test Resource");

        // List
        let all = catalog.list_all().await;
        assert_eq!(all.len(), 1);

        // Delete
        let removed = catalog.remove_resource("urn:ids:resource:test").await;
        assert!(removed);
        assert_eq!(catalog.count().await, 0);
    }

    #[tokio::test]
    async fn test_catalog_search() {
        let catalog = ResourceCatalog::new();

        catalog
            .add_resource(
                DataResource::new(
                    IdsUri::new("urn:ids:resource:1").expect("valid"),
                    "Temperature Sensor Data",
                    test_publisher(),
                )
                .with_keyword("sensor")
                .with_keyword("temperature"),
            )
            .await;

        catalog
            .add_resource(
                DataResource::new(
                    IdsUri::new("urn:ids:resource:2").expect("valid"),
                    "Traffic Data",
                    test_publisher(),
                )
                .with_keyword("traffic"),
            )
            .await;

        let results = catalog.search("temperature").await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Temperature Sensor Data");

        let results = catalog.search("sensor").await;
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_resource_builder() {
        let resource = DataResource::new(test_uri(), "My Resource", test_publisher())
            .with_description("Description")
            .with_keyword("key1")
            .with_keyword("key2")
            .with_content_type("application/json")
            .with_language("en")
            .with_byte_size(1024);

        assert_eq!(resource.title, "My Resource");
        assert_eq!(resource.keywords.len(), 2);
        assert_eq!(resource.content_type, Some("application/json".to_string()));
        assert_eq!(resource.language, Some("en".to_string()));
        assert_eq!(resource.byte_size, Some(1024));
    }

    #[test]
    fn test_distribution_builder() {
        let dist = Distribution::new("https://example.org/data", "JSON", "application/json")
            .with_byte_size(1024)
            .with_checksum("sha256:abc123");

        assert_eq!(dist.access_url, "https://example.org/data");
        assert_eq!(dist.format, "JSON");
        assert_eq!(dist.byte_size, Some(1024));
    }
}
