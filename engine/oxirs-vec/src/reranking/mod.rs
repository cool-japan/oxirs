//! Re-ranking with cross-encoders for improved search quality
//!
//! This module implements two-stage retrieval:
//! 1. Fast retrieval with bi-encoders (semantic search)
//! 2. Precise re-ranking with cross-encoders
//!
//! Cross-encoders jointly encode query and document for more accurate
//! relevance scoring, but at higher computational cost. By applying them
//! only to top-k candidates, we get both speed and accuracy.
//!
//! ## Features
//! - Multiple cross-encoder backends (local models, API-based)
//! - Score fusion strategies (linear, rank-based, learned)
//! - Batch processing for efficiency
//! - Result caching
//! - Diversity-aware re-ranking
//!
//! ## Example
//! ```rust,ignore
//! use oxirs_vec::reranking::{CrossEncoderReranker, RerankingConfig};
//!
//! let config = RerankingConfig::default();
//! let reranker = CrossEncoderReranker::new(config)?;
//!
//! // First stage: fast retrieval
//! let candidates = vector_store.search("machine learning", 100)?;
//!
//! // Second stage: precise re-ranking
//! let reranked = reranker.rerank("machine learning", &candidates, 10)?;
//! ```

pub mod cache;
pub mod config;
pub mod cross_encoder;
pub mod diversity;
pub mod fusion;
pub mod models;
pub mod reranker;
pub mod types;

pub use cache::{RerankingCache, RerankingCacheConfig};
pub use config::{FusionStrategy, RerankingConfig, RerankingMode};
pub use cross_encoder::{CrossEncoder, CrossEncoderBackend};
pub use diversity::{DiversityReranker, DiversityStrategy};
pub use fusion::{ScoreFusion, ScoreFusionConfig};
pub use models::{CrossEncoderModel, ModelBackend, ModelConfig};
pub use reranker::{CrossEncoderReranker, RerankingOutput, RerankingStats};
pub use types::{RerankingError, RerankingResult as Result, ScoredCandidate};
