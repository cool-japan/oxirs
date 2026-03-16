//! Cloud provider integration for embedding services
//!
//! This module provides comprehensive integration with major cloud providers
//! including AWS SageMaker, Azure ML, and Google Cloud AI Platform for
//! scalable embedding deployment and inference.

pub mod awssagemakerservice_traits;
pub mod azuremlservice_traits;
pub mod cloudintegrationconfig_traits;
pub mod functions;
pub mod types;

// Re-export all public types from types module
pub use types::*;
