//! HTTP request handlers for the API
//!
//! This module contains all the endpoint handlers organized by functionality.

pub mod embeddings;
pub mod models;
pub mod predictions;
pub mod scoring;
pub mod system;

// Re-export all handlers for easy access
pub use embeddings::*;
pub use models::*;
pub use predictions::*;
pub use scoring::*;
pub use system::*;
