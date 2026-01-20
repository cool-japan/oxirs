//! International Data Spaces (IDS) Connector
//!
//! IDSA Reference Architecture Model 4.x implementation for data sovereignty.
//! Enables participation in European data spaces (Catena-X, Gaia-X).
//!
//! # Features
//!
//! - **ODRL Policy Enforcement**: Usage control with ODRL 2.2 policies
//! - **Contract Negotiation**: Automated contract lifecycle management
//! - **Data Lineage**: W3C PROV-O based provenance tracking
//! - **Data Residency**: Regional data placement and GDPR compliance
//! - **Trust Framework**: Gaia-X Self-Description integration
//! - **DAPS Integration**: Dynamic Attribute Provisioning Service
//!
//! # Standards Compliance
//!
//! - IDS Reference Architecture Model 4.x
//! - IDS Information Model
//! - ODRL 2.2 (Open Digital Rights Language)
//! - W3C PROV-O (Provenance Ontology)
//! - W3C Verifiable Credentials
//! - Gaia-X Trust Framework
//! - GDPR Articles 44-49 (International data transfers)
//!
//! # Module Organization
//!
//! - `connector` - Core IDS Connector implementation
//! - `catalog` - DCAT-AP resource catalog
//! - `contract` - Contract negotiation and lifecycle
//! - `policy` - ODRL policy engine and usage control
//! - `identity` - DAPS, Verifiable Credentials, Gaia-X registry
//! - `lineage` - Data provenance tracking (W3C PROV-O)
//! - `residency` - Data residency and GDPR compliance
//! - `message` - IDS message protocol

pub mod api;
pub mod broker;
pub mod catalog;
pub mod connector;
pub mod contract;
pub mod data_plane;
pub mod identity;
pub mod lineage;
pub mod message;
pub mod policy;
pub mod residency;
pub mod types;

pub use api::{ids_router, IdsApiState};
pub use broker::{
    BrokerCatalog, BrokerClient, BrokerResource, CatalogQuery, ConnectorEndpoint,
    ConnectorSelfDescription, MultiBrokerManager, RegistrationResult,
};
pub use catalog::{DataResource, ResourceCatalog};
pub use connector::{IdsConnector, IdsConnectorConfig};
pub use contract::{ContractNegotiator, ContractState, DataContract};
pub use data_plane::{
    DataPlaneManager, StreamTransferAdapter, TransferProcess, TransferRequest, TransferResult,
    TransferStatus, TransferType,
};
pub use identity::{DapsClient, IdentityProvider};
pub use lineage::{LineageRecord, ProvenanceGraph};
pub use policy::{OdrlPolicy, PolicyEngine, UsageController};
pub use residency::{ResidencyEnforcer, ResidencyPolicy};
pub use types::{IdsError, IdsResult, IdsUri};

/// IDS Connector version
pub const IDS_VERSION: &str = "4.2.7";

/// IDS Information Model version
pub const IDS_INFOMODEL_VERSION: &str = "4.2.7";

/// Default DAPS endpoint
pub const DEFAULT_DAPS_URL: &str = "https://daps.aisec.fraunhofer.de";

/// Default IDS broker
pub const DEFAULT_IDS_BROKER: &str = "https://broker.ids.isst.fraunhofer.de";

/// Maximum contract negotiation rounds
pub const MAX_NEGOTIATION_ROUNDS: u32 = 10;

/// Default contract validity period (days)
pub const DEFAULT_CONTRACT_VALIDITY_DAYS: i64 = 90;
