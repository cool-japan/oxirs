//! Hybrid search combining keyword and semantic vector search
//!
//! This module provides functionality to combine traditional keyword search
//! (BM25, TF-IDF) with semantic vector similarity search for improved
//! relevance and recall.

pub mod config;
pub mod fusion;
pub mod keyword;
pub mod manager;
pub mod query_expansion;
#[cfg(test)]
pub mod tests;
pub mod types;

pub use config::{HybridSearchConfig, KeywordAlgorithm, RankFusionStrategy, SearchMode};
pub use fusion::RankFusion;
pub use keyword::{Bm25Scorer, KeywordSearcher, TfidfScorer};
pub use manager::HybridSearchManager;
pub use query_expansion::QueryExpander;
pub use types::{DocumentScore, HybridQuery, HybridResult, KeywordMatch, SearchWeights};
