//! Advanced Federated Query Optimization for SPARQL 1.2 — thin facade module.
//!
//! This module historically implemented sophisticated federated query
//! processing with SERVICE clause optimization, remote endpoint discovery and
//! health monitoring, query decomposition, distributed planning, cost-based
//! optimization, parallel execution strategies, and intelligent result
//! merging.
//!
//! The implementation now lives in dedicated sibling modules and is surfaced
//! here through re-exports so existing consumers can keep importing from
//! `crate::federated_query_optimizer::*`:
//!
//! - type definitions: [`federated_query_optimizer_types`](crate::federated_query_optimizer_types)
//!   (backed by [`federated_query_types`](crate::federated_query_types))
//! - query planning: [`federated_query_optimizer_planner`](crate::federated_query_optimizer_planner)
//!   (backed by [`federated_query_planner`](crate::federated_query_planner))
//! - execution & result merging: [`federated_query_optimizer_exec`](crate::federated_query_optimizer_exec)
//!   (backed by [`federated_query_executor`](crate::federated_query_executor))

pub use crate::federated_query_optimizer_exec::*;
pub use crate::federated_query_optimizer_planner::*;
pub use crate::federated_query_optimizer_types::*;
