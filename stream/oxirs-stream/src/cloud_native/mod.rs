//! # Cloud-Native Integration Module
//!
//! Comprehensive Kubernetes and service mesh integration for OxiRS Stream,
//! providing enterprise-grade cloud-native deployment, scaling, and management capabilities.
//!
//! This module provides:
//! - Kubernetes Custom Resource Definitions (CRDs) and Operators
//! - Service mesh integration (Istio, Linkerd, Consul Connect)
//! - Auto-scaling with custom metrics
//! - Health checks and observability
//! - Multi-cloud deployment strategies
//! - GitOps integration and CI/CD pipelines
//!
//! ## Module Organization
//!
//! - `config` - Main configuration structure
//! - `kubernetes` - Kubernetes integration types
//! - `service_mesh` - Service mesh configuration
//! - `auto_scaling` - Auto-scaling configuration
//! - `observability` - Monitoring, logging, and alerting
//! - `multi_cloud` - Multi-cloud deployment
//! - `gitops` - GitOps and CI/CD pipelines
//! - `manager` - CloudNativeManager implementation

pub mod auto_scaling;
pub mod config;
pub mod gitops;
pub mod kubernetes;
pub mod manager;
pub mod multi_cloud;
pub mod observability;
pub mod service_mesh;

// Re-export main types
pub use config::CloudNativeConfig;
pub use manager::CloudNativeManager;

// Re-export configuration types for convenience
pub use auto_scaling::*;
pub use gitops::*;
pub use kubernetes::*;
pub use multi_cloud::*;
pub use observability::*;
pub use service_mesh::*;
