//! HTTP request handlers for the API
//!
//! This module contains all the endpoint handlers organized by functionality.

pub mod embeddings;
pub mod scoring;
pub mod predictions;
pub mod models;
pub mod system;

// Re-export all handlers for easy access
pub use embeddings::*;
pub use scoring::*;
pub use predictions::*;
pub use models::*;
pub use system::*;