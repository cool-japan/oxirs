//! Score fusion strategies

use crate::reranking::config::FusionStrategy;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreFusion {
    strategy: FusionStrategy,
    retrieval_weight: f32,
}

impl ScoreFusion {
    pub fn new(strategy: FusionStrategy, retrieval_weight: f32) -> Self {
        Self {
            strategy,
            retrieval_weight,
        }
    }

    pub fn fuse(&self, retrieval_score: f32, reranking_score: f32) -> f32 {
        match self.strategy {
            FusionStrategy::RerankingOnly => reranking_score,
            FusionStrategy::RetrievalOnly => retrieval_score,
            FusionStrategy::Linear => {
                self.retrieval_weight * retrieval_score
                    + (1.0 - self.retrieval_weight) * reranking_score
            }
            FusionStrategy::Harmonic => {
                2.0 / (1.0 / retrieval_score.max(0.001) + 1.0 / reranking_score.max(0.001))
            }
            FusionStrategy::Geometric => (retrieval_score * reranking_score).sqrt(),
            _ => reranking_score, // Default to reranking
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreFusionConfig {
    pub strategy: FusionStrategy,
    pub retrieval_weight: f32,
}
