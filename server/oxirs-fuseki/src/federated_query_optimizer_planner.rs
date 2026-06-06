//! # Federated Query Optimizer — Query Planning
//!
//! Re-exports all planner implementations from `federated_query_planner`:
//! `EndpointRegistry`, `EndpointDiscovery`, `QueryPlanner`,
//! `JoinOrderOptimizer`, `JoinCostModel`, `CostEstimator`,
//! `CardinalityEstimator`, `FederationStatistics`.

pub use crate::federated_query_planner::*;
