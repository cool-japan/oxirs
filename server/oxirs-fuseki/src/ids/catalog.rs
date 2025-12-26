//! IDS Resource Catalog
//!
//! DCAT-AP (Data Catalog Vocabulary - Application Profile) implementation

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
    pub fn new() -> Self {
        Self {
            resources: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn add_resource(&self, resource: DataResource) {
        self.resources
            .write()
            .await
            .insert(resource.id.as_str().to_string(), resource);
    }

    pub async fn get_resource(&self, id: &str) -> Option<DataResource> {
        self.resources.read().await.get(id).cloned()
    }

    pub async fn list_resources(&self) -> Vec<DataResource> {
        self.resources.read().await.values().cloned().collect()
    }
}

/// Data Resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataResource {
    pub id: IdsUri,
    pub title: String,
    pub description: Option<String>,
    pub keywords: Vec<String>,
    pub publisher: IdsUri,
    pub distributions: Vec<Distribution>,
    pub created_at: DateTime<Utc>,
    pub modified_at: DateTime<Utc>,
}

/// Distribution (access method)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Distribution {
    pub access_url: String,
    pub format: String,
    pub media_type: String,
}
