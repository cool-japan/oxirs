//! Endpoint health monitoring for federated SPARQL query processing.
//!
//! This module tracks the availability, latency distribution, and reliability
//! of remote SPARQL endpoints via a rolling-window probe model.
//!
//! # Overview
//!
//! The [`EndpointHealthMonitor`] collects [`ProbeResult`]s and derives a
//! per-endpoint [`EndpointStatus`] based on recent success rates and latency.
//! Operators can use this information to prefer healthy endpoints, avoid
//! degraded ones, and alert on unhealthy services.

pub mod endpoint_monitor;

pub use endpoint_monitor::{
    EndpointHealthMonitor, EndpointHealthWindow, EndpointStatus, ProbeResult,
};
