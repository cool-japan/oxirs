//! Integration with oxirs-core RDF Store
//!
//! This module provides a hybrid store that automatically routes data between:
//! - RDF storage (for semantic metadata, provenance, relationships)
//! - Time-series storage (for high-frequency numerical sensor data)
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │   HybridStore (Store trait)     │
//! └─────────────┬───────────────────┘
//!               │
//!       ┌───────┴───────┐
//!       │               │
//!       v               v
//! ┌──────────┐   ┌─────────────┐
//! │ RdfStore │   │ TsdbEngine  │
//! │ (oxirs-  │   │ (oxirs-tsdb)│
//! │  core)   │   │             │
//! └──────────┘   └─────────────┘
//! ```
//!
//! ## Routing Logic
//!
//! Data is automatically routed based on predicates:
//!
//! - **QUDT numericValue** → Time-series storage
//! - **SOSA hasSimpleResult** → Time-series storage
//! - **Custom ts:value** → Time-series storage
//! - **Everything else** → RDF storage
//!
//! ## Example
//!
//! ```rust
//! use oxirs_tsdb::integration::HybridStore;
//! use oxirs_core::model::{NamedNode, Literal, Triple};
//! use oxirs_core::rdf_store::Store;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let store = HybridStore::new()?;
//!
//! // This goes to RDF storage (metadata)
//! let sensor_uri = NamedNode::new("http://example.org/sensor1")?;
//! let rdf_type = NamedNode::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")?;
//! let sensor_class = NamedNode::new("http://www.w3.org/ns/sosa/Sensor")?;
//! store.insert_triple(Triple::new(sensor_uri.clone(), rdf_type, sensor_class))?;
//!
//! // This goes to time-series storage (numerical data)
//! let numeric_value = NamedNode::new("http://qudt.org/schema/qudt/numericValue")?;
//! let value = Literal::new("42.5");
//! store.insert_triple(Triple::new(sensor_uri, numeric_value, value))?;
//! # Ok(())
//! # }
//! ```

pub mod rdf_bridge;
pub mod store_adapter;

pub use rdf_bridge::{Confidence, DetectionResult, RdfBridge};
pub use store_adapter::HybridStore;
