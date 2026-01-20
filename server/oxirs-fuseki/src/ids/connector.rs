//! IDS Connector Core Implementation
//!
//! Main IDS Connector implementing IDSA Reference Architecture

use super::catalog::ResourceCatalog;
use super::contract::{ContractManager, InMemoryNegotiator};
use super::identity::IdentityProvider;
use super::lineage::ProvenanceGraph;
use super::policy::PolicyEngine;
use super::residency::ResidencyEnforcer;
use super::types::{IdsResult, IdsUri, Party, SecurityProfile, TransferProtocol};
use std::sync::Arc;

/// IDS Connector Configuration
#[derive(Debug, Clone)]
pub struct IdsConnectorConfig {
    /// Connector ID
    pub connector_id: IdsUri,

    /// Connector title
    pub title: String,

    /// Connector description
    pub description: String,

    /// Curator (responsible party)
    pub curator: IdsUri,

    /// Maintainer
    pub maintainer: IdsUri,

    /// Security profile
    pub security_profile: SecurityProfile,

    /// DAPS URL
    pub daps_url: String,

    /// IDS Broker URLs
    pub broker_urls: Vec<String>,

    /// Gaia-X registry URL
    pub gaiax_registry_url: Option<String>,

    /// Supported transfer protocols
    pub supported_protocols: Vec<TransferProtocol>,
}

impl Default for IdsConnectorConfig {
    fn default() -> Self {
        Self {
            connector_id: IdsUri::new("urn:ids:connector:oxirs").unwrap(),
            title: "OxiRS IDS Connector".to_string(),
            description: "Semantic Web Data Space Connector".to_string(),
            curator: IdsUri::new("https://oxirs.io").unwrap(),
            maintainer: IdsUri::new("https://oxirs.io").unwrap(),
            security_profile: SecurityProfile::TrustSecurityProfile,
            daps_url: super::DEFAULT_DAPS_URL.to_string(),
            broker_urls: vec![super::DEFAULT_IDS_BROKER.to_string()],
            gaiax_registry_url: None,
            supported_protocols: vec![TransferProtocol::Https, TransferProtocol::MultipartFormData],
        }
    }
}

/// IDS Connector
pub struct IdsConnector {
    config: IdsConnectorConfig,
    identity: Arc<IdentityProvider>,
    catalog: Arc<ResourceCatalog>,
    contract_manager: Arc<ContractManager>,
    policy_engine: Arc<PolicyEngine>,
    lineage_tracker: Arc<ProvenanceGraph>,
    residency_enforcer: Arc<ResidencyEnforcer>,
}

impl IdsConnector {
    /// Create a new IDS Connector
    pub fn new(config: IdsConnectorConfig) -> Self {
        let identity = Arc::new(IdentityProvider::new(config.daps_url.clone()));
        let catalog = Arc::new(ResourceCatalog::new());
        let policy_engine = Arc::new(PolicyEngine::new());
        let lineage_tracker = Arc::new(ProvenanceGraph::default());
        let residency_enforcer = Arc::new(ResidencyEnforcer::new());

        let negotiator = Arc::new(InMemoryNegotiator::new());
        let contract_manager = Arc::new(ContractManager::new(negotiator));

        Self {
            config,
            identity,
            catalog,
            contract_manager,
            policy_engine,
            lineage_tracker,
            residency_enforcer,
        }
    }

    /// Get connector ID
    pub fn connector_id(&self) -> &IdsUri {
        &self.config.connector_id
    }

    /// Get catalog
    pub fn catalog(&self) -> Arc<ResourceCatalog> {
        Arc::clone(&self.catalog)
    }

    /// Get contract manager
    pub fn contract_manager(&self) -> Arc<ContractManager> {
        Arc::clone(&self.contract_manager)
    }

    /// Get policy engine
    pub fn policy_engine(&self) -> Arc<PolicyEngine> {
        Arc::clone(&self.policy_engine)
    }

    /// Get lineage tracker
    pub fn lineage_tracker(&self) -> Arc<ProvenanceGraph> {
        Arc::clone(&self.lineage_tracker)
    }

    /// Get residency enforcer
    pub fn residency_enforcer(&self) -> Arc<ResidencyEnforcer> {
        Arc::clone(&self.residency_enforcer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connector_creation() {
        let config = IdsConnectorConfig::default();
        let connector = IdsConnector::new(config);

        assert_eq!(connector.connector_id().as_str(), "urn:ids:connector:oxirs");
    }
}
