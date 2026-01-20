// Copyright (c) 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: MIT OR Apache-2.0

//! AI-Powered GraphQL Features
//!
//! This module provides advanced AI capabilities for GraphQL query processing,
//! including natural language query generation, schema suggestions, anomaly
//! detection, and semantic optimization.

pub mod anomaly_detection;
pub mod natural_language_query;
pub mod performance_prediction;
pub mod schema_suggestions;
pub mod semantic_optimizer;

pub use anomaly_detection::*;
pub use natural_language_query::*;
pub use performance_prediction::*;
pub use schema_suggestions::*;
pub use semantic_optimizer::*;
