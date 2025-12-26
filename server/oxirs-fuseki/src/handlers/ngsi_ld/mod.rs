//! NGSI-LD API Implementation
//!
//! ETSI GS CIM 009 V1.6.1 compliant API for OxiRS
//! <https://www.etsi.org/deliver/etsi_gs/CIM/001_099/009/01.06.01_60/gs_CIM009v010601p.pdf>
//!
//! This module provides NGSI-LD v1.6 API endpoints compatible with:
//! - FIWARE Context Broker (Orion-LD, Scorpio, Stellio)
//! - Japan PLATEAU Smart City Platform
//! - EU Smart City initiatives
//!
//! # API Endpoints
//!
//! ## Entity Operations
//! - `POST /ngsi-ld/v1/entities` - Create entity
//! - `GET /ngsi-ld/v1/entities` - Query entities
//! - `GET /ngsi-ld/v1/entities/:id` - Get entity by ID
//! - `PATCH /ngsi-ld/v1/entities/:id/attrs` - Update entity attributes
//! - `DELETE /ngsi-ld/v1/entities/:id` - Delete entity
//!
//! ## Subscription Operations
//! - `POST /ngsi-ld/v1/subscriptions` - Create subscription
//! - `GET /ngsi-ld/v1/subscriptions` - List subscriptions
//! - `GET /ngsi-ld/v1/subscriptions/:id` - Get subscription
//! - `PATCH /ngsi-ld/v1/subscriptions/:id` - Update subscription
//! - `DELETE /ngsi-ld/v1/subscriptions/:id` - Delete subscription
//!
//! ## Batch Operations
//! - `POST /ngsi-ld/v1/entityOperations/create` - Batch create
//! - `POST /ngsi-ld/v1/entityOperations/upsert` - Batch upsert
//! - `POST /ngsi-ld/v1/entityOperations/update` - Batch update
//! - `POST /ngsi-ld/v1/entityOperations/delete` - Batch delete
//!
//! ## Temporal Operations
//! - `POST /ngsi-ld/v1/temporal/entities` - Create temporal entity
//! - `GET /ngsi-ld/v1/temporal/entities` - Query temporal entities
//! - `GET /ngsi-ld/v1/temporal/entities/:id` - Get temporal entity
//! - `DELETE /ngsi-ld/v1/temporal/entities/:id` - Delete temporal entity

pub mod batch;
pub mod content_neg;
pub mod context;
pub mod converter;
pub mod entities;
pub mod query;
pub mod server_handlers;
pub mod subscriptions;
pub mod temporal;
pub mod types;

// Re-export commonly used types
pub use types::{
    BatchOperationResult, GeoProperty, NgsiAttribute, NgsiContext, NgsiEntity, NgsiError,
    NgsiProperty, NgsiQueryParams, NgsiRelationship, NgsiSubscription,
};

pub use batch::{
    batch_create_entities, batch_delete_entities, batch_update_entities, batch_upsert_entities,
};
pub use content_neg::{NgsiContentNegotiator, NgsiFormat};
pub use converter::{NgsiRdfConverter, NgsiToRdf, RdfToNgsi};
pub use entities::{
    append_entity_attrs, create_entity, delete_entity, get_entity, query_entities,
    update_entity_attrs,
};
pub use server_handlers::*;
pub use subscriptions::{
    create_subscription, delete_subscription, get_subscription, list_subscriptions,
    update_subscription,
};
pub use temporal::{
    create_temporal_entity, delete_temporal_entity, get_temporal_entity, query_temporal_entities,
};

/// NGSI-LD API version
pub const NGSI_LD_VERSION: &str = "1.6.1";

/// Default NGSI-LD Core Context URI
pub const NGSI_LD_CORE_CONTEXT: &str =
    "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld";

/// NGSI-LD API base path
pub const NGSI_LD_BASE_PATH: &str = "/ngsi-ld/v1";

/// NGSI-LD namespace
pub const NGSI_LD_NS: &str = "https://uri.etsi.org/ngsi-ld/";

/// NGSI-LD default tenant
pub const NGSI_LD_DEFAULT_TENANT: &str = "default";

/// Supported content types for NGSI-LD
pub const NGSI_LD_CONTENT_TYPES: &[&str] = &[
    "application/ld+json",
    "application/json",
    "application/geo+json",
];

/// Maximum entities per batch operation
pub const MAX_BATCH_SIZE: usize = 1000;

/// Maximum attributes per entity
pub const MAX_ATTRIBUTES_PER_ENTITY: usize = 100;

/// Maximum subscriptions per tenant
pub const MAX_SUBSCRIPTIONS_PER_TENANT: usize = 10000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(NGSI_LD_VERSION, "1.6.1");
        assert!(NGSI_LD_CORE_CONTEXT.contains("ngsi-ld"));
        assert_eq!(NGSI_LD_BASE_PATH, "/ngsi-ld/v1");
    }
}
