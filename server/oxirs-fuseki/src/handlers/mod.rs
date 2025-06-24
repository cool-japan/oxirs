//! HTTP request handlers for SPARQL protocol and server management

pub mod sparql;
pub mod graph;
pub mod admin;
pub mod auth;

// Re-export commonly used handlers
pub use sparql::{query_handler, update_handler};
pub use admin::{ui_handler};