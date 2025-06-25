//! Evaluation metrics and benchmarking for embedding models

use crate::EmbeddingModel;
use anyhow::{anyhow, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tracing::{debug, info};

/// Comprehensive evaluation suite for knowledge graph embeddings
pub struct EvaluationSuite {
    test_triples: Vec<(String, String, String)>,
    validation_triples: Vec<(String, String, String)>,
    negative_samples: Vec<(String, String, String)>,
    config: EvaluationConfig,
}

/// Configuration for evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Number of top-k predictions to evaluate
    pub k_values: Vec<usize>,
    /// Whether to use filtered ranking (exclude known positives)
    pub use_filtered_ranking: bool,
    /// Number of negative samples per positive triple
    pub negative_sample_ratio: usize,
    /// Use parallel processing for evaluation
    pub parallel_evaluation: bool,
    /// Evaluation metrics to compute
    pub metrics: Vec<EvaluationMetric>,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            k_values: vec![1, 3, 5, 10],
            use_filtered_ranking: true,
            negative_sample_ratio: 100,
            parallel_evaluation: true,
            metrics: vec![
                EvaluationMetric::MeanRank,
                EvaluationMetric::MeanReciprocalRank,
                EvaluationMetric::HitsAtK(1),
                EvaluationMetric::HitsAtK(3),
                EvaluationMetric::HitsAtK(10),
            ],
        }
    }
}

/// Types of evaluation metrics
#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationMetric {
    MeanRank,
    MeanReciprocalRank,
    HitsAtK(usize),
    NDCG(usize),
    AveragePrecision,
    F1Score,
}

/// Evaluation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub mean_rank: f64,
    pub mean_reciprocal_rank: f64,
    pub hits_at_k: HashMap<usize, f64>,
    pub ndcg_at_k: HashMap<usize, f64>,
    pub average_precision: f64,
    pub f1_score: f64,
    pub num_test_triples: usize,
    pub evaluation_time_seconds: f64,
    pub detailed_results: Vec<TripleEvaluationResult>,
}

/// Detailed results for individual triple evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TripleEvaluationResult {
    pub triple: (String, String, String),
    pub rank: usize,
    pub score: f64,
    pub reciprocal_rank: f64,
}

impl EvaluationSuite {
    /// Create a new evaluation suite
    pub fn new(
        test_triples: Vec<(String, String, String)>,
        validation_triples: Vec<(String, String, String)>,
    ) -> Self {
        Self {
            test_triples,
            validation_triples,
            negative_samples: Vec::new(),
            config: EvaluationConfig::default(),
        }
    }
    
    /// Configure evaluation parameters
    pub fn with_config(mut self, config: EvaluationConfig) -> Self {
        self.config = config;
        self
    }
    
    /// Generate negative samples for evaluation
    pub fn generate_negative_samples(&mut self, model: &dyn EmbeddingModel) -> Result<()> {
        let entities = model.get_entities();
        let relations = model.get_relations();
        
        if entities.is_empty() || relations.is_empty() {
            return Err(anyhow!("Model has no entities or relations"));
        }
        
        let positive_set: HashSet<_> = self.test_triples.iter()
            .chain(self.validation_triples.iter())
            .collect();
        
        let mut negative_samples = Vec::new();
        let mut rng = rand::thread_rng();
        
        for positive_triple in &self.test_triples {
            let mut negatives_for_triple = 0;
            let max_attempts = self.config.negative_sample_ratio * 10;
            let mut attempts = 0;
            
            while negatives_for_triple < self.config.negative_sample_ratio && attempts < max_attempts {
                attempts += 1;
                
                // Corrupt either subject or object (not predicate)
                let corrupt_subject = rand::random::<bool>();
                
                let negative_triple = if corrupt_subject {
                    let random_entity = &entities[rand::random::<usize>() % entities.len()];
                    (random_entity.clone(), positive_triple.1.clone(), positive_triple.2.clone())
                } else {
                    let random_entity = &entities[rand::random::<usize>() % entities.len()];
                    (positive_triple.0.clone(), positive_triple.1.clone(), random_entity.clone())
                };
                
                // Make sure it's actually negative
                if !positive_set.contains(&negative_triple) {
                    negative_samples.push(negative_triple);
                    negatives_for_triple += 1;
                }
            }
        }
        
        self.negative_samples = negative_samples;
        info!("Generated {} negative samples for evaluation", self.negative_samples.len());
        
        Ok(())
    }
    
    /// Run comprehensive evaluation
    pub fn evaluate(&self, model: &dyn EmbeddingModel) -> Result<EvaluationResults> {
        let start_time = std::time::Instant::now();
        info!("Starting comprehensive model evaluation");
        
        if self.test_triples.is_empty() {
            return Err(anyhow!("No test triples available for evaluation"));
        }
        
        let detailed_results = if self.config.parallel_evaluation {
            self.evaluate_parallel(model)?
        } else {
            self.evaluate_sequential(model)?
        };
        
        let results = self.compute_aggregate_metrics(&detailed_results);
        let evaluation_time = start_time.elapsed().as_secs_f64();
        
        info!("Evaluation completed in {:.2} seconds", evaluation_time);
        info!("Mean Rank: {:.2}", results.mean_rank);
        info!("Mean Reciprocal Rank: {:.4}", results.mean_reciprocal_rank);
        
        for (k, hits) in &results.hits_at_k {
            info!("Hits@{}: {:.4}", k, hits);
        }
        
        Ok(EvaluationResults {
            evaluation_time_seconds: evaluation_time,
            detailed_results,
            ..results
        })
    }
    
    /// Evaluate model performance in parallel
    fn evaluate_parallel(&self, model: &dyn EmbeddingModel) -> Result<Vec<TripleEvaluationResult>> {
        self.test_triples
            .par_iter()
            .map(|triple| self.evaluate_triple(model, triple))
            .collect()
    }
    
    /// Evaluate model performance sequentially
    fn evaluate_sequential(&self, model: &dyn EmbeddingModel) -> Result<Vec<TripleEvaluationResult>> {
        self.test_triples
            .iter()
            .map(|triple| self.evaluate_triple(model, triple))
            .collect()
    }
    
    /// Evaluate a single triple
    fn evaluate_triple(&self, model: &dyn EmbeddingModel, triple: &(String, String, String)) -> Result<TripleEvaluationResult> {
        let (subject, predicate, object) = triple;
        
        // Score the positive triple
        let positive_score = model.score_triple(subject, predicate, object)?;
        
        // Generate candidates for ranking
        let candidates = if self.config.use_filtered_ranking {
            self.generate_filtered_candidates(model, triple)?
        } else {
            self.generate_unfiltered_candidates(model, triple)?
        };
        
        // Rank candidates
        let mut scored_candidates: Vec<_> = candidates
            .into_iter()
            .map(|(s, p, o)| {
                let score = model.score_triple(&s, &p, &o).unwrap_or(f64::NEG_INFINITY);
                ((s, p, o), score)
            })
            .collect();
        
        scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Find rank of positive triple
        let rank = scored_candidates
            .iter()
            .position(|((s, p, o), _)| s == subject && p == predicate && o == object)
            .map(|pos| pos + 1) // Convert to 1-indexed
            .unwrap_or(scored_candidates.len() + 1);
        
        Ok(TripleEvaluationResult {
            triple: triple.clone(),
            rank,
            score: positive_score,
            reciprocal_rank: 1.0 / rank as f64,
        })
    }
    
    /// Generate filtered candidates (excluding known positives)
    fn generate_filtered_candidates(&self, model: &dyn EmbeddingModel, triple: &(String, String, String)) -> Result<Vec<(String, String, String)>> {
        let entities = model.get_entities();
        let (subject, predicate, object) = triple;
        
        let known_positives: HashSet<_> = self.test_triples.iter()
            .chain(self.validation_triples.iter())
            .collect();
        
        let mut candidates = Vec::new();
        
        // Generate candidates by replacing object
        for entity in &entities {
            let candidate = (subject.clone(), predicate.clone(), entity.clone());
            if !known_positives.contains(&candidate) || candidate == *triple {
                candidates.push(candidate);
            }
        }
        
        // Generate candidates by replacing subject
        for entity in &entities {
            let candidate = (entity.clone(), predicate.clone(), object.clone());
            if !known_positives.contains(&candidate) || candidate == *triple {
                candidates.push(candidate);
            }
        }
        
        Ok(candidates)
    }
    
    /// Generate unfiltered candidates
    fn generate_unfiltered_candidates(&self, model: &dyn EmbeddingModel, triple: &(String, String, String)) -> Result<Vec<(String, String, String)>> {
        let entities = model.get_entities();
        let (subject, predicate, object) = triple;
        
        let mut candidates = Vec::new();
        
        // Generate candidates by replacing object
        for entity in &entities {
            candidates.push((subject.clone(), predicate.clone(), entity.clone()));
        }
        
        // Generate candidates by replacing subject
        for entity in &entities {
            candidates.push((entity.clone(), predicate.clone(), object.clone()));
        }
        
        Ok(candidates)
    }
    
    /// Compute aggregate metrics from detailed results
    fn compute_aggregate_metrics(&self, results: &[TripleEvaluationResult]) -> EvaluationResults {
        if results.is_empty() {
            return EvaluationResults {
                mean_rank: 0.0,
                mean_reciprocal_rank: 0.0,
                hits_at_k: HashMap::new(),
                ndcg_at_k: HashMap::new(),
                average_precision: 0.0,
                f1_score: 0.0,
                num_test_triples: 0,
                evaluation_time_seconds: 0.0,
                detailed_results: Vec::new(),
            };
        }
        
        // Mean Rank
        let mean_rank = results.iter().map(|r| r.rank as f64).sum::<f64>() / results.len() as f64;
        
        // Mean Reciprocal Rank
        let mean_reciprocal_rank = results.iter().map(|r| r.reciprocal_rank).sum::<f64>() / results.len() as f64;
        
        // Hits@K
        let mut hits_at_k = HashMap::new();
        for &k in &self.config.k_values {
            let hits = results.iter().filter(|r| r.rank <= k).count() as f64 / results.len() as f64;
            hits_at_k.insert(k, hits);
        }
        
        // NDCG@K (simplified implementation)
        let mut ndcg_at_k = HashMap::new();
        for &k in &self.config.k_values {
            let ndcg = self.compute_ndcg(results, k);
            ndcg_at_k.insert(k, ndcg);
        }
        
        // Average Precision (simplified)
        let average_precision = results.iter().map(|r| r.reciprocal_rank).sum::<f64>() / results.len() as f64;
        
        // F1 Score (using Hits@1 as precision/recall approximation)
        let hits_at_1 = hits_at_k.get(&1).copied().unwrap_or(0.0);
        let f1_score = 2.0 * hits_at_1 * hits_at_1 / (hits_at_1 + hits_at_1 + 1e-10);
        
        EvaluationResults {
            mean_rank,
            mean_reciprocal_rank,
            hits_at_k,
            ndcg_at_k,
            average_precision,
            f1_score,
            num_test_triples: results.len(),
            evaluation_time_seconds: 0.0, // Will be set by caller
            detailed_results: results.to_vec(),
        }
    }
    
    /// Compute NDCG@K (simplified implementation)
    fn compute_ndcg(&self, results: &[TripleEvaluationResult], k: usize) -> f64 {
        // Simplified NDCG calculation
        // In practice, this would use proper relevance scores
        let dcg: f64 = results
            .iter()
            .filter(|r| r.rank <= k)
            .map(|r| 1.0 / (r.rank as f64).log2())
            .sum();
        
        let idcg = 1.0; // Ideal DCG for binary relevance
        
        if idcg > 0.0 {
            dcg / idcg
        } else {
            0.0
        }
    }
}

/// Benchmark suite for comparing multiple models
pub struct BenchmarkSuite {
    evaluations: HashMap<String, EvaluationResults>,
    datasets: Vec<String>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            evaluations: HashMap::new(),
            datasets: Vec::new(),
        }
    }
    
    /// Add evaluation results for a model
    pub fn add_evaluation(&mut self, model_name: String, results: EvaluationResults) {
        self.evaluations.insert(model_name, results);
    }
    
    /// Generate comparison report
    pub fn generate_report(&self) -> BenchmarkReport {
        let mut comparisons = Vec::new();
        
        for (model_name, results) in &self.evaluations {
            comparisons.push(ModelComparison {
                model_name: model_name.clone(),
                mean_rank: results.mean_rank,
                mean_reciprocal_rank: results.mean_reciprocal_rank,
                hits_at_1: results.hits_at_k.get(&1).copied().unwrap_or(0.0),
                hits_at_10: results.hits_at_k.get(&10).copied().unwrap_or(0.0),
                evaluation_time: results.evaluation_time_seconds,
            });
        }
        
        // Sort by MRR (higher is better)
        comparisons.sort_by(|a, b| b.mean_reciprocal_rank.partial_cmp(&a.mean_reciprocal_rank).unwrap());
        
        let best_model = comparisons.first().map(|c| c.model_name.clone());
        
        BenchmarkReport {
            comparisons,
            best_model,
            num_models: self.evaluations.len(),
        }
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark report comparing multiple models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub comparisons: Vec<ModelComparison>,
    pub best_model: Option<String>,
    pub num_models: usize,
}

/// Comparison data for a single model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_name: String,
    pub mean_rank: f64,
    pub mean_reciprocal_rank: f64,
    pub hits_at_1: f64,
    pub hits_at_10: f64,
    pub evaluation_time: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TransE;
    use crate::ModelConfig;
    
    #[test]
    fn test_evaluation_suite() {
        let test_triples = vec![
            ("alice".to_string(), "knows".to_string(), "bob".to_string()),
            ("bob".to_string(), "knows".to_string(), "charlie".to_string()),
        ];
        
        let validation_triples = vec![
            ("alice".to_string(), "likes".to_string(), "bob".to_string()),
        ];
        
        let suite = EvaluationSuite::new(test_triples, validation_triples);
        assert_eq!(suite.test_triples.len(), 2);
        assert_eq!(suite.validation_triples.len(), 1);
    }
    
    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();
        
        let results1 = EvaluationResults {
            mean_rank: 10.0,
            mean_reciprocal_rank: 0.5,
            hits_at_k: [(1, 0.3), (10, 0.8)].iter().cloned().collect(),
            ndcg_at_k: HashMap::new(),
            average_precision: 0.4,
            f1_score: 0.35,
            num_test_triples: 100,
            evaluation_time_seconds: 5.0,
            detailed_results: Vec::new(),
        };
        
        suite.add_evaluation("TransE".to_string(), results1);
        
        let report = suite.generate_report();
        assert_eq!(report.num_models, 1);
        assert_eq!(report.best_model, Some("TransE".to_string()));
    }
}