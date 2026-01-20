//! OPC UA Backend for OxiRS Stream
//!
//! OPC UA (Unified Architecture) protocol support for industrial automation.
//! Compatible with:
//! - PLC systems (Siemens, Schneider, ABB, etc.)
//! - SCADA systems
//! - Industrial IoT gateways
//! - Manufacturing Execution Systems (MES)

pub mod client;
pub mod node_mapping;
pub mod subscription;
pub mod types;

pub use client::{OpcUaBackend, OpcUaClient};
pub use node_mapping::NodeMapper;
pub use subscription::SubscriptionManager;
pub use types::{
    MessageSecurityMode, NodeSubscription, OpcUaConfig, OpcUaDataChange, OpcUaStats, OpcUaValue,
    SecurityPolicy, UserIdentity,
};
