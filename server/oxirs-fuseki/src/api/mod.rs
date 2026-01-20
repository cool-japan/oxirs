//! API modules for OxiRS Fuseki
//!
//! This module contains REST API endpoints for various management functions.

pub mod rebac_management;

// Re-export commonly used types
pub use rebac_management::{
    batch_create_relationships, batch_delete_relationships, check_relationship,
    create_relationship, delete_relationship, list_object_relationships, list_relationships,
    list_subject_relationships,
};
