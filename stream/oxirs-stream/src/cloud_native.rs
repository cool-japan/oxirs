//! # Cloud-Native Integration Module
//!
//! Comprehensive Kubernetes and service mesh integration for OxiRS Stream,
//! providing enterprise-grade cloud-native deployment, scaling, and management capabilities.
//!
//! ## Refactored Module Structure
//!
//! This module has been refactored from a single 2796-line file into a well-organized
//! modular structure. See `cloud_native` submodule for implementation details.

mod cloud_native;

// Re-export everything from the modular implementation
pub use cloud_native::*;
