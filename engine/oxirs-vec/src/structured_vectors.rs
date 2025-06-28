//! Structured vector types for enhanced vector representations.
//!
//! This module provides advanced vector types including:
//! - Named dimension vectors for interpretable embeddings
//! - Hierarchical vectors for multi-level representations
//! - Temporal vectors with timestamp support
//! - Weighted dimension vectors for importance scoring
//! - Confidence-scored vectors for uncertainty modeling

use std::collections::HashMap;
use std::time::SystemTime;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{Vector, VectorData, VectorPrecision};

/// Named dimension vector where each dimension has a semantic name
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NamedDimensionVector {
    /// Mapping from dimension names to indices
    pub dimension_names: HashMap<String, usize>,
    /// Underlying vector data
    pub vector: Vector,
}

impl NamedDimensionVector {
    /// Create a new named dimension vector
    pub fn new(dimension_names: Vec<String>, values: Vec<f32>) -> Result<Self> {
        if dimension_names.len() != values.len() {
            return Err(anyhow::anyhow!("Dimension names must match values length"));
        }

        let mut name_map = HashMap::new();
        for (idx, name) in dimension_names.iter().enumerate() {
            if name_map.contains_key(name) {
                return Err(anyhow::anyhow!("Duplicate dimension name: {}", name));
            }
            name_map.insert(name.clone(), idx);
        }

        Ok(Self {
            dimension_names: name_map,
            vector: Vector::new(values),
        })
    }

    /// Get value by dimension name
    pub fn get_by_name(&self, name: &str) -> Option<f32> {
        self.dimension_names
            .get(name)
            .and_then(|&idx| match &self.vector.values {
                VectorData::F32(values) => values.get(idx).copied(),
                _ => {
                    let f32_values = self.vector.as_f32();
                    f32_values.get(idx).copied()
                }
            })
    }

    /// Set value by dimension name
    pub fn set_by_name(&mut self, name: &str, value: f32) -> Result<()> {
        if let Some(&idx) = self.dimension_names.get(name) {
            match &mut self.vector.values {
                VectorData::F32(values) => {
                    if idx < values.len() {
                        values[idx] = value;
                        Ok(())
                    } else {
                        Err(anyhow::anyhow!("Index out of bounds"))
                    }
                }
                _ => Err(anyhow::anyhow!(
                    "Vector type must be F32 for direct modification"
                )),
            }
        } else {
            Err(anyhow::anyhow!("Unknown dimension name: {}", name))
        }
    }

    /// Get dimension names in order
    pub fn dimension_names_ordered(&self) -> Vec<String> {
        let mut names: Vec<(String, usize)> = self
            .dimension_names
            .iter()
            .map(|(name, &idx)| (name.clone(), idx))
            .collect();
        names.sort_by_key(|(_, idx)| *idx);
        names.into_iter().map(|(name, _)| name).collect()
    }
}

/// Hierarchical vector with multiple levels of embeddings
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchicalVector {
    /// Hierarchy levels from coarse to fine
    pub levels: Vec<Vector>,
    /// Level names/descriptions
    pub level_names: Vec<String>,
    /// Metadata for each level
    pub level_metadata: Vec<HashMap<String, String>>,
}

impl HierarchicalVector {
    /// Create a new hierarchical vector
    pub fn new(levels: Vec<Vector>, level_names: Vec<String>) -> Result<Self> {
        if levels.len() != level_names.len() {
            return Err(anyhow::anyhow!("Levels and names must have same length"));
        }

        if levels.is_empty() {
            return Err(anyhow::anyhow!("Must have at least one level"));
        }

        let level_metadata = vec![HashMap::new(); levels.len()];

        Ok(Self {
            levels,
            level_names,
            level_metadata,
        })
    }

    /// Get vector at specific level
    pub fn get_level(&self, level: usize) -> Option<&Vector> {
        self.levels.get(level)
    }

    /// Get vector by level name
    pub fn get_level_by_name(&self, name: &str) -> Option<&Vector> {
        self.level_names
            .iter()
            .position(|n| n == name)
            .and_then(|idx| self.levels.get(idx))
    }

    /// Add metadata to a level
    pub fn add_level_metadata(&mut self, level: usize, key: String, value: String) -> Result<()> {
        if level >= self.levels.len() {
            return Err(anyhow::anyhow!("Level index out of bounds"));
        }
        self.level_metadata[level].insert(key, value);
        Ok(())
    }

    /// Compute similarity at specific level
    pub fn cosine_similarity_at_level(
        &self,
        other: &HierarchicalVector,
        level: usize,
    ) -> Result<f32> {
        let self_vec = self
            .get_level(level)
            .ok_or_else(|| anyhow::anyhow!("Level {} not found in self", level))?;
        let other_vec = other
            .get_level(level)
            .ok_or_else(|| anyhow::anyhow!("Level {} not found in other", level))?;

        self_vec.cosine_similarity(other_vec)
    }

    /// Compute weighted similarity across all levels
    pub fn weighted_similarity(&self, other: &HierarchicalVector, weights: &[f32]) -> Result<f32> {
        if self.levels.len() != other.levels.len() {
            return Err(anyhow::anyhow!(
                "Hierarchical vectors must have same number of levels"
            ));
        }

        if weights.len() != self.levels.len() {
            return Err(anyhow::anyhow!("Weights must match number of levels"));
        }

        let mut total_similarity = 0.0;
        let mut total_weight = 0.0;

        for (i, weight) in weights.iter().enumerate() {
            if *weight > 0.0 {
                let sim = self.cosine_similarity_at_level(other, i)?;
                total_similarity += sim * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            Ok(total_similarity / total_weight)
        } else {
            Ok(0.0)
        }
    }
}

/// Temporal vector with timestamp information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalVector {
    /// The vector value
    pub vector: Vector,
    /// Timestamp when the vector was created/computed
    pub timestamp: SystemTime,
    /// Optional validity duration in seconds
    pub validity_duration: Option<u64>,
    /// Time-based decay factor (0.0 to 1.0)
    pub decay_factor: f32,
}

impl TemporalVector {
    /// Create a new temporal vector
    pub fn new(vector: Vector) -> Self {
        Self {
            vector,
            timestamp: SystemTime::now(),
            validity_duration: None,
            decay_factor: 1.0,
        }
    }

    /// Create with specific timestamp
    pub fn with_timestamp(vector: Vector, timestamp: SystemTime) -> Self {
        Self {
            vector,
            timestamp,
            validity_duration: None,
            decay_factor: 1.0,
        }
    }

    /// Set validity duration
    pub fn with_validity(mut self, duration_secs: u64) -> Self {
        self.validity_duration = Some(duration_secs);
        self
    }

    /// Set decay factor
    pub fn with_decay(mut self, decay_factor: f32) -> Self {
        self.decay_factor = decay_factor.clamp(0.0, 1.0);
        self
    }

    /// Check if vector is still valid
    pub fn is_valid(&self) -> bool {
        if let Some(duration) = self.validity_duration {
            if let Ok(elapsed) = self.timestamp.elapsed() {
                return elapsed.as_secs() < duration;
            }
        }
        true
    }

    /// Get time-decayed similarity
    pub fn decayed_similarity(&self, other: &TemporalVector) -> Result<f32> {
        let base_similarity = self.vector.cosine_similarity(&other.vector)?;

        // Apply time decay based on age difference
        let self_age = self.timestamp.elapsed().unwrap_or_default().as_secs_f32();
        let other_age = other.timestamp.elapsed().unwrap_or_default().as_secs_f32();
        let age_diff = (self_age - other_age).abs();

        // Exponential decay based on age difference
        let decay = (-age_diff * (1.0 - self.decay_factor) / 3600.0).exp(); // Hourly decay

        Ok(base_similarity * decay)
    }
}

/// Weighted dimension vector where each dimension has an importance weight
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WeightedDimensionVector {
    /// The vector values
    pub vector: Vector,
    /// Importance weights for each dimension
    pub weights: Vec<f32>,
}

impl WeightedDimensionVector {
    /// Create a new weighted dimension vector
    pub fn new(values: Vec<f32>, weights: Vec<f32>) -> Result<Self> {
        if values.len() != weights.len() {
            return Err(anyhow::anyhow!("Values and weights must have same length"));
        }

        // Validate weights are non-negative
        if weights.iter().any(|&w| w < 0.0) {
            return Err(anyhow::anyhow!("Weights must be non-negative"));
        }

        Ok(Self {
            vector: Vector::new(values),
            weights,
        })
    }

    /// Create with uniform weights
    pub fn uniform(values: Vec<f32>) -> Self {
        let weight = 1.0 / values.len() as f32;
        let weights = vec![weight; values.len()];
        Self {
            vector: Vector::new(values),
            weights,
        }
    }

    /// Normalize weights to sum to 1.0
    pub fn normalize_weights(&mut self) {
        let sum: f32 = self.weights.iter().sum();
        if sum > 0.0 {
            for weight in &mut self.weights {
                *weight /= sum;
            }
        }
    }

    /// Compute weighted cosine similarity
    pub fn weighted_cosine_similarity(&self, other: &WeightedDimensionVector) -> Result<f32> {
        if self.vector.dimensions != other.vector.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_values = self.vector.as_f32();
        let other_values = other.vector.as_f32();

        // Combine weights (e.g., by averaging)
        let combined_weights: Vec<f32> = self
            .weights
            .iter()
            .zip(&other.weights)
            .map(|(w1, w2)| (w1 + w2) / 2.0)
            .collect();

        let weighted_dot: f32 = self_values
            .iter()
            .zip(&other_values)
            .zip(&combined_weights)
            .map(|((a, b), w)| a * b * w)
            .sum();

        let self_magnitude: f32 = self_values
            .iter()
            .zip(&self.weights)
            .map(|(v, w)| v * v * w)
            .sum::<f32>()
            .sqrt();

        let other_magnitude: f32 = other_values
            .iter()
            .zip(&other.weights)
            .map(|(v, w)| v * v * w)
            .sum::<f32>()
            .sqrt();

        if self_magnitude == 0.0 || other_magnitude == 0.0 {
            return Ok(0.0);
        }

        Ok(weighted_dot / (self_magnitude * other_magnitude))
    }

    /// Get the most important dimensions
    pub fn top_dimensions(&self, k: usize) -> Vec<(usize, f32, f32)> {
        let mut indexed: Vec<(usize, f32, f32)> = self
            .vector
            .as_f32()
            .iter()
            .zip(&self.weights)
            .enumerate()
            .map(|(idx, (&value, &weight))| (idx, value, weight))
            .collect();

        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }
}

/// Confidence-scored vector with uncertainty estimates
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceScoredVector {
    /// The mean vector values
    pub mean: Vector,
    /// Confidence scores or standard deviations for each dimension
    pub confidence: Vec<f32>,
    /// Overall confidence score (0.0 to 1.0)
    pub overall_confidence: f32,
}

impl ConfidenceScoredVector {
    /// Create a new confidence-scored vector
    pub fn new(mean_values: Vec<f32>, confidence_scores: Vec<f32>) -> Result<Self> {
        if mean_values.len() != confidence_scores.len() {
            return Err(anyhow::anyhow!(
                "Mean values and confidence scores must have same length"
            ));
        }

        // Validate confidence scores
        if confidence_scores.iter().any(|&c| c < 0.0 || c > 1.0) {
            return Err(anyhow::anyhow!(
                "Confidence scores must be between 0.0 and 1.0"
            ));
        }

        let overall_confidence =
            confidence_scores.iter().sum::<f32>() / confidence_scores.len() as f32;

        Ok(Self {
            mean: Vector::new(mean_values),
            confidence: confidence_scores,
            overall_confidence,
        })
    }

    /// Create with uniform high confidence
    pub fn high_confidence(values: Vec<f32>) -> Self {
        let confidence = vec![0.95; values.len()];
        Self {
            mean: Vector::new(values),
            overall_confidence: 0.95,
            confidence,
        }
    }

    /// Compute similarity with confidence weighting
    pub fn confidence_weighted_similarity(&self, other: &ConfidenceScoredVector) -> Result<f32> {
        if self.mean.dimensions != other.mean.dimensions {
            return Err(anyhow::anyhow!("Vector dimensions must match"));
        }

        let self_values = self.mean.as_f32();
        let other_values = other.mean.as_f32();

        // Use confidence scores as weights
        let weighted_dot: f32 = self_values
            .iter()
            .zip(&other_values)
            .zip(self.confidence.iter().zip(&other.confidence))
            .map(|((a, b), (c1, c2))| a * b * c1 * c2)
            .sum();

        let self_magnitude: f32 = self_values
            .iter()
            .zip(&self.confidence)
            .map(|(v, c)| v * v * c)
            .sum::<f32>()
            .sqrt();

        let other_magnitude: f32 = other_values
            .iter()
            .zip(&other.confidence)
            .map(|(v, c)| v * v * c)
            .sum::<f32>()
            .sqrt();

        if self_magnitude == 0.0 || other_magnitude == 0.0 {
            return Ok(0.0);
        }

        let similarity = weighted_dot / (self_magnitude * other_magnitude);

        // Scale by overall confidence
        Ok(similarity * self.overall_confidence * other.overall_confidence)
    }

    /// Sample vector from confidence distribution (assuming Gaussian)
    pub fn sample(&self) -> Vector {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = thread_rng();
        let values = self.mean.as_f32();
        let mut sampled = Vec::new();

        for (i, &mean_val) in values.iter().enumerate() {
            let std_dev = (1.0 - self.confidence[i]) * mean_val.abs() * 0.1; // Convert confidence to std dev
            if std_dev > 0.0 {
                let normal = Normal::new(mean_val, std_dev).unwrap();
                sampled.push(normal.sample(&mut rng));
            } else {
                sampled.push(mean_val);
            }
        }

        Vector::new(sampled)
    }

    /// Get dimensions with low confidence
    pub fn low_confidence_dimensions(&self, threshold: f32) -> Vec<(usize, f32, f32)> {
        self.mean
            .as_f32()
            .iter()
            .zip(&self.confidence)
            .enumerate()
            .filter(|(_, (_, &conf))| conf < threshold)
            .map(|(idx, (&value, &conf))| (idx, value, conf))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_named_dimension_vector() {
        let names = vec!["age".to_string(), "income".to_string(), "score".to_string()];
        let values = vec![25.0, 50000.0, 0.85];

        let mut named_vec = NamedDimensionVector::new(names, values).unwrap();

        assert_eq!(named_vec.get_by_name("age"), Some(25.0));
        assert_eq!(named_vec.get_by_name("income"), Some(50000.0));
        assert_eq!(named_vec.get_by_name("unknown"), None);

        named_vec.set_by_name("score", 0.95).unwrap();
        assert_eq!(named_vec.get_by_name("score"), Some(0.95));
    }

    #[test]
    fn test_hierarchical_vector() {
        let level1 = Vector::new(vec![1.0, 2.0]);
        let level2 = Vector::new(vec![1.0, 2.0, 3.0, 4.0]);
        let level3 = Vector::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let levels = vec![level1, level2, level3];
        let names = vec![
            "coarse".to_string(),
            "medium".to_string(),
            "fine".to_string(),
        ];

        let hier_vec = HierarchicalVector::new(levels, names).unwrap();

        assert_eq!(hier_vec.levels.len(), 3);
        assert!(hier_vec.get_level_by_name("medium").is_some());
        assert_eq!(hier_vec.get_level_by_name("medium").unwrap().dimensions, 4);
    }

    #[test]
    fn test_temporal_vector() {
        let vec = Vector::new(vec![1.0, 2.0, 3.0]);
        let temporal = TemporalVector::new(vec)
            .with_validity(3600) // 1 hour
            .with_decay(0.9);

        assert!(temporal.is_valid());
        assert_eq!(temporal.decay_factor, 0.9);
    }

    #[test]
    fn test_weighted_dimension_vector() {
        let values = vec![1.0, 2.0, 3.0];
        let weights = vec![0.1, 0.3, 0.6];

        let mut weighted = WeightedDimensionVector::new(values, weights).unwrap();
        weighted.normalize_weights();

        let sum: f32 = weighted.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        let top = weighted.top_dimensions(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, 2); // Index of highest weight
    }

    #[test]
    fn test_confidence_scored_vector() {
        let values = vec![1.0, 2.0, 3.0];
        let confidence = vec![0.9, 0.8, 0.95];

        let conf_vec = ConfidenceScoredVector::new(values, confidence).unwrap();

        assert!(conf_vec.overall_confidence > 0.8);

        let low_conf = conf_vec.low_confidence_dimensions(0.85);
        assert_eq!(low_conf.len(), 1);
        assert_eq!(low_conf[0].0, 1); // Index with 0.8 confidence
    }
}
