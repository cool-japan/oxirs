//! Tenant representation and management

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Unique identifier for a tenant
pub type TenantId = String;

/// Tenant status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TenantStatus {
    /// Tenant is active and operational
    Active,

    /// Tenant is suspended (temporary)
    Suspended,

    /// Tenant is in trial period
    Trial,

    /// Tenant is being provisioned
    Provisioning,

    /// Tenant is being decommissioned
    Decommissioning,

    /// Tenant has been deleted
    Deleted,
}

impl TenantStatus {
    /// Check if tenant can perform operations
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Active | Self::Trial)
    }

    /// Check if tenant is in a terminal state
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Deleted)
    }
}

/// Tenant metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantMetadata {
    /// Tenant display name
    pub name: String,

    /// Organization/company name
    pub organization: Option<String>,

    /// Contact email
    pub email: Option<String>,

    /// Tenant description
    pub description: Option<String>,

    /// Billing contact
    pub billing_contact: Option<String>,

    /// Technical contact
    pub technical_contact: Option<String>,

    /// Region/datacenter location
    pub region: Option<String>,

    /// Custom tags for organization
    pub tags: HashMap<String, String>,

    /// Tenant tier (e.g., "free", "pro", "enterprise")
    pub tier: String,
}

impl TenantMetadata {
    /// Create new tenant metadata
    pub fn new(name: impl Into<String>, tier: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            organization: None,
            email: None,
            description: None,
            billing_contact: None,
            technical_contact: None,
            region: None,
            tags: HashMap::new(),
            tier: tier.into(),
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Get a tag value
    pub fn get_tag(&self, key: &str) -> Option<&String> {
        self.tags.get(key)
    }
}

/// Represents a tenant in the multi-tenant system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tenant {
    /// Unique tenant identifier
    pub id: TenantId,

    /// Tenant metadata
    pub metadata: TenantMetadata,

    /// Current tenant status
    pub status: TenantStatus,

    /// Tenant creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last updated timestamp
    pub updated_at: DateTime<Utc>,

    /// Trial expiration (if applicable)
    pub trial_expires_at: Option<DateTime<Utc>>,

    /// Namespace prefix for data isolation
    pub namespace: String,

    /// Custom configuration for the tenant
    pub config: HashMap<String, String>,
}

impl Tenant {
    /// Create a new tenant
    pub fn new(id: impl Into<String>, metadata: TenantMetadata) -> Self {
        let id = id.into();
        let namespace = Self::generate_namespace(&id);

        Self {
            id,
            metadata,
            status: TenantStatus::Active,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            trial_expires_at: None,
            namespace,
            config: HashMap::new(),
        }
    }

    /// Create a new tenant with auto-generated ID
    pub fn new_with_auto_id(metadata: TenantMetadata) -> Self {
        let id = Uuid::new_v4().to_string();
        Self::new(id, metadata)
    }

    /// Create a tenant in trial mode
    pub fn new_trial(id: impl Into<String>, metadata: TenantMetadata, trial_days: u32) -> Self {
        let mut tenant = Self::new(id, metadata);
        tenant.status = TenantStatus::Trial;
        tenant.trial_expires_at = Some(Utc::now() + chrono::Duration::days(trial_days as i64));
        tenant
    }

    /// Generate namespace from tenant ID
    fn generate_namespace(id: &str) -> String {
        format!("tenant_{}", id.replace('-', "_"))
    }

    /// Check if tenant is active and operational
    pub fn is_operational(&self) -> bool {
        self.status.is_operational()
    }

    /// Check if trial has expired
    pub fn is_trial_expired(&self) -> bool {
        if let Some(expires_at) = self.trial_expires_at {
            Utc::now() > expires_at
        } else {
            false
        }
    }

    /// Update tenant status
    pub fn set_status(&mut self, status: TenantStatus) {
        self.status = status;
        self.updated_at = Utc::now();
    }

    /// Suspend the tenant
    pub fn suspend(&mut self) {
        self.set_status(TenantStatus::Suspended);
    }

    /// Activate the tenant
    pub fn activate(&mut self) {
        self.set_status(TenantStatus::Active);
    }

    /// Convert trial to paid
    pub fn convert_trial_to_paid(&mut self, new_tier: impl Into<String>) {
        if self.status == TenantStatus::Trial {
            self.status = TenantStatus::Active;
            self.trial_expires_at = None;
            self.metadata.tier = new_tier.into();
            self.updated_at = Utc::now();
        }
    }

    /// Get tenant age in days
    pub fn age_days(&self) -> i64 {
        (Utc::now() - self.created_at).num_days()
    }

    /// Set configuration value
    pub fn set_config(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.config.insert(key.into(), value.into());
        self.updated_at = Utc::now();
    }

    /// Get configuration value
    pub fn get_config(&self, key: &str) -> Option<&String> {
        self.config.get(key)
    }

    /// Get namespaced key for data isolation
    pub fn namespaced_key(&self, key: &str) -> String {
        format!("{}:{}", self.namespace, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_creation() {
        let metadata = TenantMetadata::new("Test Tenant", "pro");
        let tenant = Tenant::new("tenant1", metadata);

        assert_eq!(tenant.id, "tenant1");
        assert_eq!(tenant.metadata.name, "Test Tenant");
        assert_eq!(tenant.metadata.tier, "pro");
        assert_eq!(tenant.status, TenantStatus::Active);
        assert!(tenant.is_operational());
    }

    #[test]
    fn test_tenant_auto_id() {
        let metadata = TenantMetadata::new("Auto ID Tenant", "free");
        let tenant = Tenant::new_with_auto_id(metadata);

        assert!(!tenant.id.is_empty());
        assert_eq!(tenant.metadata.name, "Auto ID Tenant");
    }

    #[test]
    fn test_tenant_trial() {
        let metadata = TenantMetadata::new("Trial Tenant", "trial");
        let tenant = Tenant::new_trial("tenant2", metadata, 30);

        assert_eq!(tenant.status, TenantStatus::Trial);
        assert!(tenant.trial_expires_at.is_some());
        assert!(!tenant.is_trial_expired());
        assert!(tenant.is_operational());
    }

    #[test]
    fn test_tenant_status_changes() {
        let metadata = TenantMetadata::new("Test", "pro");
        let mut tenant = Tenant::new("tenant3", metadata);

        assert!(tenant.is_operational());

        tenant.suspend();
        assert_eq!(tenant.status, TenantStatus::Suspended);
        assert!(!tenant.is_operational());

        tenant.activate();
        assert_eq!(tenant.status, TenantStatus::Active);
        assert!(tenant.is_operational());
    }

    #[test]
    fn test_trial_conversion() {
        let metadata = TenantMetadata::new("Trial Convert", "trial");
        let mut tenant = Tenant::new_trial("tenant4", metadata, 30);

        assert_eq!(tenant.status, TenantStatus::Trial);
        assert_eq!(tenant.metadata.tier, "trial");
        assert!(tenant.trial_expires_at.is_some());

        tenant.convert_trial_to_paid("enterprise");

        assert_eq!(tenant.status, TenantStatus::Active);
        assert_eq!(tenant.metadata.tier, "enterprise");
        assert!(tenant.trial_expires_at.is_none());
    }

    #[test]
    fn test_tenant_config() {
        let metadata = TenantMetadata::new("Config Test", "pro");
        let mut tenant = Tenant::new("tenant5", metadata);

        tenant.set_config("max_vectors", "1000000");
        tenant.set_config("index_type", "hnsw");

        assert_eq!(
            tenant.get_config("max_vectors"),
            Some(&"1000000".to_string())
        );
        assert_eq!(tenant.get_config("index_type"), Some(&"hnsw".to_string()));
        assert_eq!(tenant.get_config("nonexistent"), None);
    }

    #[test]
    fn test_namespaced_key() {
        let metadata = TenantMetadata::new("Namespace Test", "pro");
        let tenant = Tenant::new("tenant6", metadata);

        let key = tenant.namespaced_key("vectors");
        assert!(key.contains("tenant_tenant6"));
        assert!(key.contains("vectors"));
    }

    #[test]
    fn test_tenant_metadata() {
        let mut metadata = TenantMetadata::new("Metadata Test", "enterprise");
        metadata.organization = Some("Acme Corp".to_string());
        metadata.email = Some("admin@acme.com".to_string());
        metadata.region = Some("us-west-2".to_string());
        metadata.add_tag("environment", "production");
        metadata.add_tag("cost_center", "engineering");

        assert_eq!(metadata.organization, Some("Acme Corp".to_string()));
        assert_eq!(
            metadata.get_tag("environment"),
            Some(&"production".to_string())
        );
        assert_eq!(
            metadata.get_tag("cost_center"),
            Some(&"engineering".to_string())
        );
        assert_eq!(metadata.get_tag("nonexistent"), None);
    }

    #[test]
    fn test_tenant_status_operational() {
        assert!(TenantStatus::Active.is_operational());
        assert!(TenantStatus::Trial.is_operational());
        assert!(!TenantStatus::Suspended.is_operational());
        assert!(!TenantStatus::Deleted.is_operational());
        assert!(!TenantStatus::Provisioning.is_operational());
    }

    #[test]
    fn test_tenant_status_terminal() {
        assert!(!TenantStatus::Active.is_terminal());
        assert!(!TenantStatus::Suspended.is_terminal());
        assert!(TenantStatus::Deleted.is_terminal());
    }
}
