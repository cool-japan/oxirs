//! # Federated Query Optimizer — Execution Support
//!
//! Re-exports all execution implementations from `federated_query_executor`:
//! `FederatedQueryOptimizer`, `FederatedExecutor`, `ClientPool`, `RetryPolicy`,
//! and the merge-strategy types.

pub use crate::federated_query_executor::*;
