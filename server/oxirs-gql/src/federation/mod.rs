//! GraphQL Federation and Schema Stitching Support
//!
//! This module provides advanced federation capabilities for OxiRS GraphQL, including:
//! - Dynamic service discovery and health monitoring
//! - Remote schema introspection and merging
//! - Schema composition and stitching
//! - Cross-service query planning and optimization
//! - RDF dataset federation
//! - Load balancing and failover
//! - Distributed query tracing across subgraphs
//! - Comprehensive schema validation
//! - Cross-service authentication propagation
//! - Federated subscription support
//! - Automatic schema composition

pub mod auth_propagation;
pub mod automatic_composition;
pub mod config;
pub mod dataset_federation;
pub mod distributed_tracing;
pub mod enhanced_manager;
pub mod federated_subscriptions;
pub mod manager;
pub mod query_planner;
pub mod real_time_sync;
pub mod schema_stitcher;
pub mod schema_validation;
pub mod service_discovery;

pub use auth_propagation::*;
pub use config::*;
pub use dataset_federation::*;
pub use distributed_tracing::*;
pub use enhanced_manager::*;
pub use federated_subscriptions::*;
pub use manager::*;
pub use query_planner::*;
pub use real_time_sync::*;
pub use schema_stitcher::*;
pub use service_discovery::*;

// Re-export from schema_validation to avoid conflicts with automatic_composition
pub use schema_validation::{
    ArgumentDefinition as ValidationArgumentDefinition,
    DirectiveDefinition,
    FederationSchemaValidator,
    FieldDefinition as ValidationFieldDefinition,
    SubgraphSchema as ValidationSubgraphSchema,
    // Export with prefixes to avoid conflicts
    TypeDefinition as ValidationTypeDefinition,
    TypeKind as ValidationTypeKind,
    ValidationIssue,
    ValidationLocation,
    ValidationResult as SchemaValidationResult,
    ValidationSeverity,
};

// Re-export from automatic_composition
pub use automatic_composition::{
    ArgumentDefinition as CompositionArgumentDefinition,
    AutomaticSchemaComposer,
    ComposedSchema,
    CompositionConflict,
    CompositionResult,
    FieldDefinition as CompositionFieldDefinition,
    // Export with prefixes to avoid conflicts
    SubgraphSchema as CompositionSubgraphSchema,
    TypeDefinition as CompositionTypeDefinition,
    TypeKind as CompositionTypeKind,
};
