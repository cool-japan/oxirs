//! Evaluation metrics and assessment for transformer embedding models

use super::types::EmbeddingEvaluationMetrics;
use crate::EmbeddingError;
use anyhow::Result;
use ndarray::Array1;
use std::collections::HashMap;

/// Evaluation engine for transformer embeddings
#[derive(Debug)]
pub struct TransformerEvaluator {
    entity_embeddings: HashMap<String, Array1<f32>>,
    relation_embeddings: HashMap<String, Array1<f32>>,
}

impl TransformerEvaluator {
    pub fn new(
        entity_embeddings: HashMap<String, Array1<f32>>,
        relation_embeddings: HashMap<String, Array1<f32>>,
    ) -> Self {
        Self {
            entity_embeddings,
            relation_embeddings,
        }
    }

    /// Comprehensive evaluation of embeddings
    pub fn evaluate_embeddings(&self) -> Result<EmbeddingEvaluationMetrics> {
        if self.entity_embeddings.is_empty() {
            return Err(EmbeddingError::ModelNotTrained.into());
        }

        let mut metrics = EmbeddingEvaluationMetrics::default();

        // Compute embedding quality metrics
        self.compute_embedding_coherence(&mut metrics)?;
        self.compute_similarity_distribution(&mut metrics)?;
        self.compute_clustering_quality(&mut metrics)?;

        // Knowledge graph specific metrics
        self.compute_semantic_coherence(&mut metrics)?;

        Ok(metrics)
    }

    /// Compute embedding coherence (how well embeddings capture semantic relationships)
    fn compute_embedding_coherence(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        let mut coherence_scores = Vec::new();
        let entities: Vec<_> = self.entity_embeddings.keys().collect();

        // Compare embeddings pairwise
        for (i, entity1) in entities.iter().enumerate() {
            for entity2 in entities.iter().skip(i + 1) {
                if let (Some(emb1), Some(emb2)) = (
                    self.entity_embeddings.get(*entity1),
                    self.entity_embeddings.get(*entity2),
                ) {
                    let similarity = self.cosine_similarity(emb1, emb2);
                    coherence_scores.push(similarity);
                }
            }
        }

        if !coherence_scores.is_empty() {
            metrics.coherence_score = coherence_scores.iter().sum::<f32>() / coherence_scores.len() as f32;
        }

        Ok(())
    }

    /// Compute similarity distribution to measure embedding diversity
    fn compute_similarity_distribution(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        let mut similarities = Vec::new();
        let entities: Vec<_> = self.entity_embeddings.values().collect();

        for (i, emb1) in entities.iter().enumerate() {
            for emb2 in entities.iter().skip(i + 1) {
                let similarity = self.cosine_similarity(emb1, emb2);
                similarities.push(similarity);
            }
        }

        if !similarities.is_empty() {
            // Compute diversity as variance of similarities
            let mean = similarities.iter().sum::<f32>() / similarities.len() as f32;
            let variance = similarities
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / similarities.len() as f32;

            metrics.diversity_score = variance;
            metrics.mean_cosine_similarity = mean;
            metrics.embedding_variance = variance;
        }

        Ok(())
    }

    /// Compute clustering quality using silhouette analysis
    fn compute_clustering_quality(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        // Simple clustering quality based on embedding distribution
        let embeddings: Vec<_> = self.entity_embeddings.values().collect();
        
        if embeddings.len() < 2 {
            return Ok(());
        }

        let mut intra_cluster_distances = Vec::new();
        let mut inter_cluster_distances = Vec::new();

        // Simplified clustering: assume entities with similar first dimension values are in same cluster
        for (i, emb1) in embeddings.iter().enumerate() {
            for (j, emb2) in embeddings.iter().enumerate() {
                if i != j {
                    let distance = self.euclidean_distance(emb1, emb2);
                    
                    // Simple heuristic: same cluster if first dimension values are close
                    if (emb1[0] - emb2[0]).abs() < 0.5 {
                        intra_cluster_distances.push(distance);
                    } else {
                        inter_cluster_distances.push(distance);
                    }
                }
            }
        }

        // Compute silhouette-like score
        let avg_intra = if !intra_cluster_distances.is_empty() {
            intra_cluster_distances.iter().sum::<f32>() / intra_cluster_distances.len() as f32
        } else {
            0.0
        };

        let avg_inter = if !inter_cluster_distances.is_empty() {
            inter_cluster_distances.iter().sum::<f32>() / inter_cluster_distances.len() as f32
        } else {
            0.0
        };

        if avg_inter > 0.0 {
            metrics.cluster_quality = (avg_inter - avg_intra) / avg_inter.max(avg_intra);
        }

        Ok(())
    }

    /// Compute semantic coherence of embeddings
    fn compute_semantic_coherence(&self, metrics: &mut EmbeddingEvaluationMetrics) -> Result<()> {
        // Measure how well embeddings preserve semantic relationships
        let mut coherence_sum = 0.0;
        let mut count = 0;

        // Check if similar entities have similar embeddings
        let entities: Vec<_> = self.entity_embeddings.keys().collect();
        
        for entity1 in &entities {
            for entity2 in &entities {
                if entity1 != entity2 {
                    if let (Some(emb1), Some(emb2)) = (
                        self.entity_embeddings.get(*entity1),
                        self.entity_embeddings.get(*entity2),
                    ) {
                        // Simple semantic similarity based on entity name similarity
                        let name_similarity = self.string_similarity(entity1, entity2);
                        let embedding_similarity = self.cosine_similarity(emb1, emb2);
                        
                        // Coherence is correlation between name and embedding similarity
                        coherence_sum += name_similarity * embedding_similarity;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            metrics.semantic_consistency = coherence_sum / count as f32;
        }

        Ok(())
    }

    /// Compute cosine similarity between two embeddings
    fn cosine_similarity(&self, emb1: &Array1<f32>, emb2: &Array1<f32>) -> f32 {
        let dot_product = (emb1 * emb2).sum();
        let norm1 = emb1.mapv(|x| x * x).sum().sqrt();
        let norm2 = emb2.mapv(|x| x * x).sum().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }

    /// Compute Euclidean distance between two embeddings
    fn euclidean_distance(&self, emb1: &Array1<f32>, emb2: &Array1<f32>) -> f32 {
        (emb1 - emb2).mapv(|x| x * x).sum().sqrt()
    }

    /// Compute string similarity (simple character-based)
    fn string_similarity(&self, s1: &str, s2: &str) -> f32 {
        let chars1: std::collections::HashSet<char> = s1.chars().collect();
        let chars2: std::collections::HashSet<char> = s2.chars().collect();

        let intersection = chars1.intersection(&chars2).count();
        let union = chars1.union(&chars2).count();

        if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        }
    }

    /// Evaluate embedding quality for specific domains
    pub fn evaluate_domain_quality(&self, domain: &str) -> Result<f32> {
        match domain {
            "scientific" => self.evaluate_scientific_quality(),
            "biomedical" => self.evaluate_biomedical_quality(),
            "legal" => self.evaluate_legal_quality(),
            "code" => self.evaluate_code_quality(),
            _ => Ok(0.5), // Default neutral score
        }
    }

    /// Evaluate quality for scientific domain
    fn evaluate_scientific_quality(&self) -> Result<f32> {
        let mut quality_score = 0.0;
        let mut count = 0;

        // Look for scientific terms and their relationships
        for (entity, embedding) in &self.entity_embeddings {
            if entity.to_lowercase().contains("dna") || 
               entity.to_lowercase().contains("protein") ||
               entity.to_lowercase().contains("molecule") {
                // Check if scientific entities have reasonable embeddings
                let norm = embedding.mapv(|x| x * x).sum().sqrt();
                if norm > 0.1 && norm < 10.0 {
                    quality_score += 1.0;
                }
                count += 1;
            }
        }

        Ok(if count > 0 { quality_score / count as f32 } else { 0.5 })
    }

    /// Evaluate quality for biomedical domain
    fn evaluate_biomedical_quality(&self) -> Result<f32> {
        let mut quality_score = 0.0;
        let mut count = 0;

        // Look for biomedical terms
        for (entity, embedding) in &self.entity_embeddings {
            if entity.to_lowercase().contains("gene") || 
               entity.to_lowercase().contains("protein") ||
               entity.to_lowercase().contains("cell") {
                let norm = embedding.mapv(|x| x * x).sum().sqrt();
                if norm > 0.1 && norm < 10.0 {
                    quality_score += 1.0;
                }
                count += 1;
            }
        }

        Ok(if count > 0 { quality_score / count as f32 } else { 0.5 })
    }

    /// Evaluate quality for legal domain
    fn evaluate_legal_quality(&self) -> Result<f32> {
        let mut quality_score = 0.0;
        let mut count = 0;

        // Look for legal terms
        for (entity, embedding) in &self.entity_embeddings {
            if entity.to_lowercase().contains("law") || 
               entity.to_lowercase().contains("court") ||
               entity.to_lowercase().contains("legal") {
                let norm = embedding.mapv(|x| x * x).sum().sqrt();
                if norm > 0.1 && norm < 10.0 {
                    quality_score += 1.0;
                }
                count += 1;
            }
        }

        Ok(if count > 0 { quality_score / count as f32 } else { 0.5 })
    }

    /// Evaluate quality for code domain
    fn evaluate_code_quality(&self) -> Result<f32> {
        let mut quality_score = 0.0;
        let mut count = 0;

        // Look for code terms
        for (entity, embedding) in &self.entity_embeddings {
            if entity.to_lowercase().contains("function") || 
               entity.to_lowercase().contains("class") ||
               entity.to_lowercase().contains("method") {
                let norm = embedding.mapv(|x| x * x).sum().sqrt();
                if norm > 0.1 && norm < 10.0 {
                    quality_score += 1.0;
                }
                count += 1;
            }
        }

        Ok(if count > 0 { quality_score / count as f32 } else { 0.5 })
    }

    /// Generate detailed evaluation report
    pub fn generate_evaluation_report(&self) -> Result<String> {
        let metrics = self.evaluate_embeddings()?;
        
        let report = format!(
            "Transformer Embedding Evaluation Report\n\
             =====================================\n\
             \n\
             Coherence Score: {:.3}\n\
             Diversity Score: {:.3}\n\
             Cluster Quality: {:.3}\n\
             Semantic Consistency: {:.3}\n\
             Mean Cosine Similarity: {:.3}\n\
             Embedding Variance: {:.3}\n\
             \n\
             Entity Count: {}\n\
             Relation Count: {}\n\
             \n\
             Quality Assessment:\n\
             - Coherence: {}\n\
             - Diversity: {}\n\
             - Clustering: {}\n\
             - Semantics: {}\n",
            metrics.coherence_score,
            metrics.diversity_score,
            metrics.cluster_quality,
            metrics.semantic_consistency,
            metrics.mean_cosine_similarity,
            metrics.embedding_variance,
            self.entity_embeddings.len(),
            self.relation_embeddings.len(),
            self.assess_quality_level(metrics.coherence_score),
            self.assess_quality_level(metrics.diversity_score),
            self.assess_quality_level(metrics.cluster_quality),
            self.assess_quality_level(metrics.semantic_consistency),
        );

        Ok(report)
    }

    /// Assess quality level based on score
    fn assess_quality_level(&self, score: f32) -> &'static str {
        match score {
            s if s > 0.8 => "Excellent",
            s if s > 0.6 => "Good",
            s if s > 0.4 => "Fair",
            s if s > 0.2 => "Poor",
            _ => "Very Poor",
        }
    }

    /// Get embedding statistics
    pub fn get_embedding_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();

        // Entity embedding statistics
        if !self.entity_embeddings.is_empty() {
            let mut norms = Vec::new();
            let mut means = Vec::new();

            for embedding in self.entity_embeddings.values() {
                norms.push(embedding.mapv(|x| x * x).sum().sqrt());
                means.push(embedding.mean().unwrap_or(0.0));
            }

            stats.insert("entity_avg_norm".to_string(), 
                         norms.iter().sum::<f32>() / norms.len() as f32);
            stats.insert("entity_avg_mean".to_string(), 
                         means.iter().sum::<f32>() / means.len() as f32);
        }

        // Relation embedding statistics
        if !self.relation_embeddings.is_empty() {
            let mut norms = Vec::new();
            let mut means = Vec::new();

            for embedding in self.relation_embeddings.values() {
                norms.push(embedding.mapv(|x| x * x).sum().sqrt());
                means.push(embedding.mean().unwrap_or(0.0));
            }

            stats.insert("relation_avg_norm".to_string(), 
                         norms.iter().sum::<f32>() / norms.len() as f32);
            stats.insert("relation_avg_mean".to_string(), 
                         means.iter().sum::<f32>() / means.len() as f32);
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_evaluator_creation() {
        let mut entity_embeddings = HashMap::new();
        entity_embeddings.insert("entity1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]));
        entity_embeddings.insert("entity2".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0]));

        let relation_embeddings = HashMap::new();
        let evaluator = TransformerEvaluator::new(entity_embeddings, relation_embeddings);

        assert_eq!(evaluator.entity_embeddings.len(), 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let entity_embeddings = HashMap::new();
        let relation_embeddings = HashMap::new();
        let evaluator = TransformerEvaluator::new(entity_embeddings, relation_embeddings);

        let emb1 = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let emb2 = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let emb3 = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        assert_eq!(evaluator.cosine_similarity(&emb1, &emb2), 0.0);
        assert_eq!(evaluator.cosine_similarity(&emb1, &emb3), 1.0);
    }

    #[test]
    fn test_domain_quality_evaluation() {
        let mut entity_embeddings = HashMap::new();
        entity_embeddings.insert("dna_sequence".to_string(), Array1::from_vec(vec![1.0, 0.5, 0.2]));
        entity_embeddings.insert("protein_structure".to_string(), Array1::from_vec(vec![0.8, 0.6, 0.1]));

        let relation_embeddings = HashMap::new();
        let evaluator = TransformerEvaluator::new(entity_embeddings, relation_embeddings);

        let scientific_quality = evaluator.evaluate_domain_quality("scientific").unwrap();
        assert!(scientific_quality > 0.0);
    }

    #[test]
    fn test_evaluation_report_generation() {
        let mut entity_embeddings = HashMap::new();
        entity_embeddings.insert("entity1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0]));
        entity_embeddings.insert("entity2".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0]));

        let relation_embeddings = HashMap::new();
        let evaluator = TransformerEvaluator::new(entity_embeddings, relation_embeddings);

        let report = evaluator.generate_evaluation_report();
        assert!(report.is_ok());
        assert!(report.unwrap().contains("Evaluation Report"));
    }
}