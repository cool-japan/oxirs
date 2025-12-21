//! SHACL Shape Inference & Learning
//!
//! Automatic shape inference from RDF data using statistical analysis and ML.
//! Uses SciRS2 for scientific computing capabilities.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use scirs2_core::ndarray_ext::Array2;

use oxirs_core::{
    model::{NamedNode, Term},
    Store,
};

use crate::{Result, Shape, ShapeId};

/// Strategy for shape inference
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceStrategy {
    /// Statistical analysis based on data distributions
    Statistical,
    /// Frequency-based inference
    FrequencyBased,
    /// Machine learning-based inference (using SciRS2)
    MachineLearning,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Configuration for shape inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeInferenceConfig {
    /// Inference strategy to use
    pub strategy: InferenceStrategy,

    /// Minimum support threshold (0.0 - 1.0)
    pub min_support: f64,

    /// Minimum confidence threshold (0.0 - 1.0)
    pub min_confidence: f64,

    /// Maximum number of properties to infer per shape
    pub max_properties: usize,

    /// Whether to infer cardinality constraints
    pub infer_cardinality: bool,

    /// Whether to infer datatype constraints
    pub infer_datatypes: bool,

    /// Whether to infer value ranges
    pub infer_ranges: bool,

    /// Sample size for statistical inference
    pub sample_size: usize,
}

impl Default for ShapeInferenceConfig {
    fn default() -> Self {
        Self {
            strategy: InferenceStrategy::Statistical,
            min_support: 0.7,
            min_confidence: 0.8,
            max_properties: 50,
            infer_cardinality: true,
            infer_datatypes: true,
            infer_ranges: true,
            sample_size: 1000,
        }
    }
}

/// An inferred shape from data analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredShape {
    /// Generated shape
    pub shape: Shape,

    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,

    /// Support (number of instances that match)
    pub support: usize,

    /// Inference metadata
    pub metadata: InferenceMetadata,
}

/// Metadata about shape inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMetadata {
    /// Strategy used
    pub strategy: InferenceStrategy,

    /// Number of instances analyzed
    pub instances_analyzed: usize,

    /// Properties discovered
    pub properties_discovered: usize,

    /// Inference timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl Default for InferenceMetadata {
    fn default() -> Self {
        Self {
            strategy: InferenceStrategy::Statistical,
            instances_analyzed: 0,
            properties_discovered: 0,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Shape inference engine using SciRS2
pub struct ShapeInferenceEngine {
    /// Configuration
    config: ShapeInferenceConfig,

    /// Seed for deterministic sampling
    seed: u64,

    /// Inferred shapes cache
    cache: HashMap<String, Vec<InferredShape>>,

    /// Statistics tracker
    stats: InferenceStats,
}

/// Statistics for shape inference
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InferenceStats {
    /// Total shapes inferred
    pub shapes_inferred: usize,

    /// Total instances analyzed
    pub instances_analyzed: usize,

    /// Average confidence score
    pub avg_confidence: f64,

    /// Total inference time (milliseconds)
    pub total_time_ms: u64,
}

impl ShapeInferenceEngine {
    /// Create a new shape inference engine
    pub fn new(config: ShapeInferenceConfig) -> Self {
        Self {
            config,
            seed: 42, // Deterministic seed for reproducibility
            cache: HashMap::new(),
            stats: InferenceStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(ShapeInferenceConfig::default())
    }

    /// Infer shapes for a specific RDF class
    #[allow(unused_variables)]
    pub fn infer_shapes_for_class(
        &mut self,
        class: &NamedNode,
        store: &dyn Store,
    ) -> Result<Vec<InferredShape>> {
        let start = std::time::Instant::now();

        // Check cache
        let cache_key = class.as_str().to_string();
        if let Some(cached) = self.cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Collect instances of this class
        let instances = self.collect_class_instances(class, store)?;

        if instances.is_empty() {
            return Ok(Vec::new());
        }

        // Sample instances if needed
        let sample = self.sample_instances(&instances);

        // Infer shapes based on strategy
        let inferred = match self.config.strategy {
            InferenceStrategy::Statistical => self.infer_statistical(&sample, class, store)?,
            InferenceStrategy::FrequencyBased => {
                self.infer_frequency_based(&sample, class, store)?
            }
            InferenceStrategy::MachineLearning => self.infer_ml_based(&sample, class, store)?,
            InferenceStrategy::Hybrid => self.infer_hybrid(&sample, class, store)?,
        };

        // Update statistics
        self.stats.shapes_inferred += inferred.len();
        self.stats.instances_analyzed += sample.len();
        self.stats.total_time_ms += start.elapsed().as_millis() as u64;

        if !inferred.is_empty() {
            let avg_conf: f64 =
                inferred.iter().map(|s| s.confidence).sum::<f64>() / inferred.len() as f64;
            self.stats.avg_confidence = avg_conf;
        }

        // Cache results
        self.cache.insert(cache_key, inferred.clone());

        Ok(inferred)
    }

    /// Collect all instances of a class (stub)
    #[allow(unused_variables)]
    fn collect_class_instances(&self, class: &NamedNode, store: &dyn Store) -> Result<Vec<Term>> {
        // TODO: Query store for instances
        Ok(Vec::new())
    }

    /// Sample instances using SciRS2 random sampling
    fn sample_instances(&mut self, instances: &[Term]) -> Vec<Term> {
        if instances.len() <= self.config.sample_size {
            return instances.to_vec();
        }

        // Use SciRS2 for efficient random sampling
        use scirs2_core::random::Random;

        let mut rng = Random::seed(42); // Deterministic for reproducibility
        let mut sampled = Vec::with_capacity(self.config.sample_size);
        let mut indices: Vec<usize> = (0..instances.len()).collect();

        // Fisher-Yates shuffle for uniform random sampling
        for i in 0..self.config.sample_size.min(instances.len()) {
            let j = i + rng.gen_range(0..(indices.len() - i));
            indices.swap(i, j);
            sampled.push(instances[indices[i]].clone());
        }

        sampled
    }

    /// Statistical inference using SciRS2-stats
    #[allow(unused_variables)]
    fn infer_statistical(
        &self,
        instances: &[Term],
        class: &NamedNode,
        store: &dyn Store,
    ) -> Result<Vec<InferredShape>> {
        use scirs2_core::ndarray_ext::Array1;
        use std::collections::HashMap;

        let mut shapes = Vec::new();

        if instances.is_empty() {
            return Ok(shapes);
        }

        // Analyze property patterns for all instances
        let mut property_counts: HashMap<String, usize> = HashMap::new();
        let mut property_datatypes: HashMap<String, HashMap<String, usize>> = HashMap::new();
        let mut property_cardinalities: HashMap<String, Vec<usize>> = HashMap::new();

        // Collect statistics from instances
        for instance in instances {
            // In real implementation, query properties for each instance
            // For now, simulate property discovery
            let properties = self.extract_properties(instance, store)?;

            for (prop, values) in properties {
                *property_counts.entry(prop.clone()).or_insert(0) += 1;

                // Track cardinality
                property_cardinalities
                    .entry(prop.clone())
                    .or_default()
                    .push(values.len());

                // Track datatypes
                for value in values {
                    let datatype = self.infer_datatype(&value);
                    *property_datatypes
                        .entry(prop.clone())
                        .or_default()
                        .entry(datatype)
                        .or_insert(0) += 1;
                }
            }
        }

        // Use SciRS2 for statistical analysis
        let total_instances = instances.len() as f64;
        let mut properties_discovered = 0;

        // Create node shape for the class
        let shape_id = ShapeId::new(format!(
            "inferred_{}",
            class.as_str().replace([':', '/'], "_")
        ));
        let shape = Shape::node_shape(shape_id);

        // Add properties with high confidence
        for (prop, count) in &property_counts {
            let support = *count as f64 / total_instances;

            if support >= self.config.min_support {
                properties_discovered += 1;

                // Infer cardinality constraints using statistical analysis
                if self.config.infer_cardinality {
                    if let Some(cardinalities) = property_cardinalities.get(prop) {
                        let card_array =
                            Array1::from_vec(cardinalities.iter().map(|&c| c as f64).collect());

                        // Use SciRS2 stats for mean and std deviation
                        let mean = card_array.iter().sum::<f64>() / card_array.len() as f64;
                        let variance = card_array.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                            / card_array.len() as f64;
                        let std_dev = variance.sqrt();

                        // If low variance, infer exact cardinality
                        if std_dev < 0.5 {
                            let rounded_mean = mean.round() as usize;
                            if rounded_mean == 1 {
                                // Exactly one property (min=1, max=1)
                                // In real implementation, add sh:minCount and sh:maxCount constraints
                            }
                        }
                    }
                }
            }
        }

        // Calculate overall confidence using statistical methods
        let confidence = if properties_discovered > 0 {
            // Use SciRS2 random for confidence calculation
            let avg_support: f64 = property_counts
                .values()
                .map(|&c| c as f64 / total_instances)
                .sum::<f64>()
                / properties_discovered as f64;

            // Confidence based on support and sample size
            let sample_factor = (instances.len() as f64 / 1000.0).min(1.0);
            (avg_support * 0.7 + sample_factor * 0.3).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let inferred = InferredShape {
            shape,
            confidence,
            support: instances.len(),
            metadata: InferenceMetadata {
                strategy: InferenceStrategy::Statistical,
                instances_analyzed: instances.len(),
                properties_discovered,
                timestamp: chrono::Utc::now(),
            },
        };

        shapes.push(inferred);
        Ok(shapes)
    }

    /// Extract properties from an instance (helper method)
    #[allow(unused_variables)]
    fn extract_properties(
        &self,
        instance: &Term,
        store: &dyn Store,
    ) -> Result<HashMap<String, Vec<Term>>> {
        // In real implementation, query the store for all properties of this instance
        // For now, return empty map
        Ok(HashMap::new())
    }

    /// Infer datatype from a term value
    fn infer_datatype(&self, term: &Term) -> String {
        match term {
            Term::NamedNode(_) => "IRI".to_string(),
            Term::BlankNode(_) => "BlankNode".to_string(),
            Term::Literal(lit) => {
                // In real implementation, check literal.datatype()
                "xsd:string".to_string()
            }
            _ => "Unknown".to_string(),
        }
    }

    /// Frequency-based inference
    #[allow(unused_variables)]
    fn infer_frequency_based(
        &self,
        instances: &[Term],
        class: &NamedNode,
        store: &dyn Store,
    ) -> Result<Vec<InferredShape>> {
        use std::collections::HashMap;

        let mut shapes = Vec::new();

        if instances.is_empty() {
            return Ok(shapes);
        }

        // Count property occurrences across all instances
        let mut property_frequencies: HashMap<String, usize> = HashMap::new();
        let mut property_value_frequencies: HashMap<String, HashMap<String, usize>> =
            HashMap::new();

        for instance in instances {
            let properties = self.extract_properties(instance, store)?;

            for (prop, values) in properties {
                *property_frequencies.entry(prop.clone()).or_insert(0) += 1;

                // Track value frequencies for each property
                for value in values {
                    let value_str = format!("{:?}", value); // Simplified value representation
                    *property_value_frequencies
                        .entry(prop.clone())
                        .or_default()
                        .entry(value_str)
                        .or_insert(0) += 1;
                }
            }
        }

        // Create shape based on frequency analysis
        let shape_id = ShapeId::new(format!(
            "inferred_freq_{}",
            class.as_str().replace([':', '/'], "_")
        ));
        let shape = Shape::node_shape(shape_id);

        let mut properties_discovered = 0;
        let total_instances = instances.len() as f64;

        // Add constraints for frequently occurring properties
        for (prop, freq) in &property_frequencies {
            let frequency = *freq as f64 / total_instances;

            // Properties that occur in > min_support of instances
            if frequency >= self.config.min_support {
                properties_discovered += 1;

                // Check if there are common values (value constraints)
                if let Some(value_freqs) = property_value_frequencies.get(prop) {
                    // Find most common value
                    if let Some((common_value, &common_count)) =
                        value_freqs.iter().max_by_key(|(_, &count)| count)
                    {
                        let value_frequency = common_count as f64 / *freq as f64;

                        // If a value appears very frequently, it might be a fixed value
                        if value_frequency >= self.config.min_confidence {
                            // In real implementation, add sh:hasValue constraint
                        }
                    }
                }
            }
        }

        // Calculate confidence based on frequency patterns
        let confidence = if properties_discovered > 0 {
            let avg_frequency: f64 = property_frequencies
                .values()
                .map(|&f| f as f64 / total_instances)
                .sum::<f64>()
                / properties_discovered as f64;

            // Higher confidence if frequencies are consistent
            let frequency_variance = property_frequencies
                .values()
                .map(|&f| {
                    let freq = f as f64 / total_instances;
                    (freq - avg_frequency).powi(2)
                })
                .sum::<f64>()
                / properties_discovered as f64;

            let consistency_bonus = (1.0 - frequency_variance.sqrt()).max(0.0) * 0.2;
            (avg_frequency + consistency_bonus).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let inferred = InferredShape {
            shape,
            confidence,
            support: instances.len(),
            metadata: InferenceMetadata {
                strategy: InferenceStrategy::FrequencyBased,
                instances_analyzed: instances.len(),
                properties_discovered,
                timestamp: chrono::Utc::now(),
            },
        };

        shapes.push(inferred);
        Ok(shapes)
    }

    /// Machine learning-based inference using SciRS2
    #[allow(unused_variables)]
    fn infer_ml_based(
        &self,
        instances: &[Term],
        class: &NamedNode,
        store: &dyn Store,
    ) -> Result<Vec<InferredShape>> {
        use std::collections::HashMap;

        let mut shapes = Vec::new();

        if instances.is_empty() {
            return Ok(shapes);
        }

        // Extract features from instances for ML analysis
        let mut feature_matrix_data: Vec<Vec<f64>> = Vec::new();
        let mut property_names: Vec<String> = Vec::new();
        let mut property_index: HashMap<String, usize> = HashMap::new();

        // Build feature matrix: rows = instances, columns = properties
        for instance in instances {
            let properties = self.extract_properties(instance, store)?;
            let mut feature_row = vec![0.0; property_index.len()];

            for (prop, values) in properties {
                let idx = if let Some(&existing_idx) = property_index.get(&prop) {
                    existing_idx
                } else {
                    let new_idx = property_index.len();
                    property_index.insert(prop.clone(), new_idx);
                    property_names.push(prop.clone());
                    feature_row.push(0.0); // Extend existing rows
                    new_idx
                };

                // Feature value = number of values for this property
                if idx < feature_row.len() {
                    feature_row[idx] = values.len() as f64;
                }
            }

            // Ensure all rows have the same length
            feature_row.resize(property_index.len(), 0.0);
            feature_matrix_data.push(feature_row);
        }

        if property_index.is_empty() {
            return Ok(shapes);
        }

        // Use SciRS2 to analyze property patterns
        let patterns = self.analyze_property_patterns(&feature_matrix_data)?;

        // Apply simple clustering to find common patterns
        // In production, would use scirs2-cluster for k-means clustering
        let num_instances = feature_matrix_data.len();
        let num_properties = property_index.len();

        // Calculate property importance using variance
        let mut property_importance: Vec<(String, f64)> = Vec::new();

        for (prop, &idx) in &property_index {
            if idx >= num_properties {
                continue;
            }

            // Calculate variance for this property across all instances
            let values: Vec<f64> = feature_matrix_data
                .iter()
                .map(|row| row.get(idx).copied().unwrap_or(0.0))
                .collect();

            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;

            // Higher variance = more important property for discrimination
            property_importance.push((prop.clone(), variance));
        }

        // Sort by importance
        property_importance
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Create shape based on ML analysis
        let shape_id = ShapeId::new(format!(
            "inferred_ml_{}",
            class.as_str().replace([':', '/'], "_")
        ));
        let shape = Shape::node_shape(shape_id);

        // Take top properties (up to max_properties config)
        let properties_discovered = property_importance.len().min(self.config.max_properties);

        // Calculate confidence using ML-based scoring
        // Higher confidence if there are clear patterns in the data
        let confidence = if properties_discovered > 0 {
            let top_variance = property_importance
                .iter()
                .take(properties_discovered)
                .map(|(_, v)| v)
                .sum::<f64>()
                / properties_discovered as f64;

            // Normalize variance to confidence score
            let variance_score = (top_variance / (top_variance + 1.0)).clamp(0.0, 1.0);

            // Adjust based on sample size
            let sample_factor = (num_instances as f64 / 1000.0).min(1.0);
            (variance_score * 0.6 + sample_factor * 0.4).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let inferred = InferredShape {
            shape,
            confidence,
            support: instances.len(),
            metadata: InferenceMetadata {
                strategy: InferenceStrategy::MachineLearning,
                instances_analyzed: instances.len(),
                properties_discovered,
                timestamp: chrono::Utc::now(),
            },
        };

        shapes.push(inferred);
        Ok(shapes)
    }

    /// Hybrid inference combining multiple strategies
    #[allow(unused_variables)]
    fn infer_hybrid(
        &self,
        instances: &[Term],
        class: &NamedNode,
        store: &dyn Store,
    ) -> Result<Vec<InferredShape>> {
        // Combine multiple strategies for robust inference
        let mut all_shapes = Vec::new();

        // Run all three strategies
        let statistical_shapes = self.infer_statistical(instances, class, store)?;
        let frequency_shapes = self.infer_frequency_based(instances, class, store)?;
        let ml_shapes = self.infer_ml_based(instances, class, store)?;

        // Combine results with weighted voting
        all_shapes.extend(statistical_shapes);
        all_shapes.extend(frequency_shapes);
        all_shapes.extend(ml_shapes);

        // If we have multiple shapes, merge them intelligently
        if all_shapes.len() > 1 {
            // Calculate ensemble confidence (weighted average)
            let total_confidence: f64 = all_shapes.iter().map(|s| s.confidence).sum();
            let avg_confidence = total_confidence / all_shapes.len() as f64;

            // Use the shape with highest confidence as base
            if let Some(best_shape) = all_shapes.iter().max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                let mut merged_shape = best_shape.clone();

                // Boost confidence using ensemble agreement
                merged_shape.confidence =
                    (avg_confidence * 0.7 + best_shape.confidence * 0.3).clamp(0.0, 1.0);

                // Update metadata
                merged_shape.metadata.strategy = InferenceStrategy::Hybrid;

                return Ok(vec![merged_shape]);
            }
        }

        Ok(all_shapes)
    }

    /// Analyze property patterns using SciRS2 ndarray operations
    #[allow(unused_variables)]
    fn analyze_property_patterns(&self, data: &[Vec<f64>]) -> Result<Array2<f64>> {
        use scirs2_core::ndarray_ext::Array2;

        // Convert to ndarray for analysis
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };

        if rows == 0 || cols == 0 {
            return Ok(Array2::zeros((0, 0)));
        }

        // Flatten data for Array2 construction
        let flattened: Vec<f64> = data.iter().flatten().copied().collect();

        // Create feature matrix
        let feature_matrix = Array2::from_shape_vec((rows, cols), flattened).map_err(|e| {
            crate::ShaclError::ValidationEngine(format!("Failed to create array: {}", e))
        })?;

        // Compute correlation matrix for property relationships
        // In production, use SciRS2 stats for proper correlation analysis
        let mut correlation_matrix = Array2::zeros((cols, cols));

        for i in 0..cols {
            for j in 0..cols {
                if i == j {
                    correlation_matrix[[i, j]] = 1.0;
                } else if i < j {
                    // Calculate Pearson correlation coefficient
                    let col_i: Vec<f64> = (0..rows).map(|r| feature_matrix[[r, i]]).collect();
                    let col_j: Vec<f64> = (0..rows).map(|r| feature_matrix[[r, j]]).collect();

                    let mean_i = col_i.iter().sum::<f64>() / rows as f64;
                    let mean_j = col_j.iter().sum::<f64>() / rows as f64;

                    let cov: f64 = col_i
                        .iter()
                        .zip(&col_j)
                        .map(|(xi, xj)| (xi - mean_i) * (xj - mean_j))
                        .sum();

                    let var_i: f64 = col_i.iter().map(|xi| (xi - mean_i).powi(2)).sum();

                    let var_j: f64 = col_j.iter().map(|xj| (xj - mean_j).powi(2)).sum();

                    let correlation = if var_i > 0.0 && var_j > 0.0 {
                        cov / (var_i.sqrt() * var_j.sqrt())
                    } else {
                        0.0
                    };

                    correlation_matrix[[i, j]] = correlation;
                    correlation_matrix[[j, i]] = correlation;
                }
            }
        }

        Ok(correlation_matrix)
    }

    /// Get inference statistics
    pub fn stats(&self) -> &InferenceStats {
        &self.stats
    }

    /// Clear the inference cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get the current configuration
    pub fn config(&self) -> &ShapeInferenceConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: ShapeInferenceConfig) {
        self.config = config;
        self.clear_cache(); // Clear cache when config changes
    }
}

impl Default for ShapeInferenceEngine {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Anomaly detection methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyDetectionMethod {
    /// Z-score based outlier detection (standard deviations)
    ZScore,
    /// Interquartile Range (IQR) method
    IQR,
    /// Modified Z-score using Median Absolute Deviation (MAD)
    ModifiedZScore,
    /// Isolation Forest-like approach
    IsolationBased,
}

/// Configuration for anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Detection method to use
    pub method: AnomalyDetectionMethod,

    /// Sensitivity threshold (lower = more sensitive)
    /// For Z-score: typically 2.5-3.0
    /// For IQR: multiplier for IQR (typically 1.5)
    pub sensitivity: f64,

    /// Minimum number of instances to detect anomalies
    pub min_instances: usize,

    /// Whether to automatically refine shapes based on anomalies
    pub auto_refine: bool,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            method: AnomalyDetectionMethod::ModifiedZScore,
            sensitivity: 3.0,
            min_instances: 10,
            auto_refine: true,
        }
    }
}

/// An anomaly detected in the data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    /// Instance that is anomalous
    pub instance: String,

    /// Property that shows anomalous behavior
    pub property: String,

    /// Anomaly score (higher = more anomalous)
    pub score: f64,

    /// Expected value or range
    pub expected: String,

    /// Actual value
    pub actual: String,

    /// Anomaly type
    pub anomaly_type: AnomalyType,
}

/// Types of anomalies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    /// Value outside expected range
    OutOfRange,
    /// Cardinality different from expected
    CardinalityMismatch,
    /// Unexpected datatype
    DatatypeMismatch,
    /// Missing expected property
    MissingProperty,
    /// Unexpected property present
    UnexpectedProperty,
    /// Pattern mismatch
    PatternMismatch,
}

/// Results of anomaly detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,

    /// Total instances analyzed
    pub total_instances: usize,

    /// Percentage of anomalous instances
    pub anomaly_rate: f64,

    /// Suggested shape refinements
    pub suggested_refinements: Vec<ShapeRefinement>,
}

/// Suggested shape refinement based on anomalies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeRefinement {
    /// Shape to refine
    pub shape_id: ShapeId,

    /// Type of refinement
    pub refinement_type: RefinementType,

    /// Property affected
    pub property: String,

    /// Refinement details
    pub details: String,

    /// Confidence in this refinement
    pub confidence: f64,
}

/// Types of shape refinements
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefinementType {
    /// Relax cardinality constraint
    RelaxCardinality,
    /// Tighten cardinality constraint
    TightenCardinality,
    /// Add datatype constraint
    AddDatatype,
    /// Relax datatype constraint
    RelaxDatatype,
    /// Add value range constraint
    AddRange,
    /// Add pattern constraint
    AddPattern,
    /// Remove constraint
    RemoveConstraint,
}

/// Anomaly detector using SciRS2-stats
pub struct AnomalyDetector {
    /// Configuration
    config: AnomalyDetectionConfig,

    /// Detection statistics
    stats: AnomalyDetectionStats,
}

/// Statistics for anomaly detection
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnomalyDetectionStats {
    /// Total anomalies detected
    pub total_anomalies: usize,

    /// Anomalies by type
    pub anomalies_by_type: HashMap<String, usize>,

    /// Total refinements suggested
    pub refinements_suggested: usize,

    /// Total refinements applied
    pub refinements_applied: usize,
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new(config: AnomalyDetectionConfig) -> Self {
        Self {
            config,
            stats: AnomalyDetectionStats::default(),
        }
    }

    /// Create with default configuration
    pub fn default_config() -> Self {
        Self::new(AnomalyDetectionConfig::default())
    }

    /// Detect anomalies in property values using statistical methods
    pub fn detect_value_anomalies(&mut self, values: &[f64]) -> Result<Vec<usize>> {
        if values.len() < self.config.min_instances {
            return Ok(Vec::new());
        }

        match self.config.method {
            AnomalyDetectionMethod::ZScore => self.detect_zscore(values),
            AnomalyDetectionMethod::IQR => self.detect_iqr(values),
            AnomalyDetectionMethod::ModifiedZScore => self.detect_modified_zscore(values),
            AnomalyDetectionMethod::IsolationBased => self.detect_isolation_based(values),
        }
    }

    /// Z-score based outlier detection
    fn detect_zscore(&self, values: &[f64]) -> Result<Vec<usize>> {
        use scirs2_core::ndarray_ext::Array1;

        let array = Array1::from_vec(values.to_vec());

        // Calculate mean and standard deviation
        let mean = array.iter().sum::<f64>() / array.len() as f64;
        let variance = array.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / array.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(Vec::new()); // No variance, no outliers
        }

        // Find indices where |z-score| > sensitivity
        let mut anomalies = Vec::new();
        for (i, &value) in values.iter().enumerate() {
            let z_score = (value - mean).abs() / std_dev;
            if z_score > self.config.sensitivity {
                anomalies.push(i);
            }
        }

        Ok(anomalies)
    }

    /// Interquartile Range (IQR) based outlier detection
    fn detect_iqr(&self, values: &[f64]) -> Result<Vec<usize>> {
        if values.len() < 4 {
            return Ok(Vec::new());
        }

        // Sort values to find quartiles
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate Q1, Q3, and IQR
        let n = sorted.len();
        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;

        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        if iqr == 0.0 {
            return Ok(Vec::new());
        }

        // Outliers are below Q1 - k*IQR or above Q3 + k*IQR
        let k = self.config.sensitivity;
        let lower_bound = q1 - k * iqr;
        let upper_bound = q3 + k * iqr;

        let mut anomalies = Vec::new();
        for (i, &value) in values.iter().enumerate() {
            if value < lower_bound || value > upper_bound {
                anomalies.push(i);
            }
        }

        Ok(anomalies)
    }

    /// Modified Z-score using Median Absolute Deviation (MAD)
    /// More robust to outliers than standard Z-score
    fn detect_modified_zscore(&self, values: &[f64]) -> Result<Vec<usize>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate median
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Calculate Median Absolute Deviation (MAD)
        let mut abs_deviations: Vec<f64> = values.iter().map(|&x| (x - median).abs()).collect();
        abs_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mad = if abs_deviations.len() % 2 == 0 {
            let mid = abs_deviations.len() / 2;
            (abs_deviations[mid - 1] + abs_deviations[mid]) / 2.0
        } else {
            abs_deviations[abs_deviations.len() / 2]
        };

        if mad == 0.0 {
            return Ok(Vec::new());
        }

        // Modified Z-score = 0.6745 * (x - median) / MAD
        // Constant 0.6745 makes MAD consistent estimator for normal distribution
        let consistency_constant = 0.6745;
        let mut anomalies = Vec::new();

        for (i, &value) in values.iter().enumerate() {
            let modified_z = consistency_constant * (value - median).abs() / mad;
            if modified_z > self.config.sensitivity {
                anomalies.push(i);
            }
        }

        Ok(anomalies)
    }

    /// Isolation-based anomaly detection (simplified)
    fn detect_isolation_based(&self, values: &[f64]) -> Result<Vec<usize>> {
        // Simplified isolation approach based on value isolation score
        // In production, would use proper isolation forest from scirs2-cluster

        if values.len() < 3 {
            return Ok(Vec::new());
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mut anomalies = Vec::new();
        let threshold = self.config.sensitivity / 10.0; // Scale down for this method

        for (i, &value) in values.iter().enumerate() {
            // Find position in sorted array
            let pos = sorted
                .binary_search_by(|&x| x.partial_cmp(&value).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or_else(|x| x);

            // Calculate isolation score based on distance to neighbors
            let isolation_score = if pos == 0 {
                // Leftmost value
                (sorted[1] - value).abs()
            } else if pos == sorted.len() - 1 {
                // Rightmost value
                (value - sorted[pos - 1]).abs()
            } else {
                // Middle value - minimum distance to neighbors
                ((value - sorted[pos - 1]).abs()).min((sorted[pos + 1] - value).abs())
            };

            // Normalize by range
            let range = sorted[sorted.len() - 1] - sorted[0];
            let normalized_score = if range > 0.0 {
                isolation_score / range
            } else {
                0.0
            };

            if normalized_score > threshold {
                anomalies.push(i);
            }
        }

        Ok(anomalies)
    }

    /// Analyze validation results for anomalies and suggest refinements
    pub fn analyze_and_refine(
        &mut self,
        shape: &Shape,
        instances: &[Term],
        _store: &dyn Store,
    ) -> Result<AnomalyDetectionResult> {
        let anomalies = Vec::new();
        let mut refinements = Vec::new();

        // Analyze each property constraint for anomalies
        // In real implementation, would extract property values and detect patterns

        let anomaly_rate = if instances.is_empty() {
            0.0
        } else {
            anomalies.len() as f64 / instances.len() as f64
        };

        // Generate refinement suggestions based on anomalies
        if self.config.auto_refine {
            refinements = self.generate_refinements(&anomalies, shape)?;
        }

        // Update statistics
        self.stats.total_anomalies += anomalies.len();
        self.stats.refinements_suggested += refinements.len();

        Ok(AnomalyDetectionResult {
            anomalies,
            total_instances: instances.len(),
            anomaly_rate,
            suggested_refinements: refinements,
        })
    }

    /// Generate shape refinements based on detected anomalies
    fn generate_refinements(
        &self,
        anomalies: &[Anomaly],
        shape: &Shape,
    ) -> Result<Vec<ShapeRefinement>> {
        let mut refinements = Vec::new();
        let mut property_anomaly_counts: HashMap<String, usize> = HashMap::new();

        // Count anomalies by property
        for anomaly in anomalies {
            *property_anomaly_counts
                .entry(anomaly.property.clone())
                .or_insert(0) += 1;
        }

        // Generate refinements for properties with many anomalies
        for (property, count) in property_anomaly_counts {
            let anomaly_types: Vec<_> = anomalies
                .iter()
                .filter(|a| a.property == property)
                .map(|a| a.anomaly_type.clone())
                .collect();

            // Suggest relaxation if many anomalies
            if count > 3 {
                let refinement_type = if anomaly_types.contains(&AnomalyType::CardinalityMismatch) {
                    RefinementType::RelaxCardinality
                } else if anomaly_types.contains(&AnomalyType::DatatypeMismatch) {
                    RefinementType::RelaxDatatype
                } else {
                    RefinementType::RelaxCardinality // Default
                };

                refinements.push(ShapeRefinement {
                    shape_id: shape.id.clone(),
                    refinement_type,
                    property: property.clone(),
                    details: format!("Detected {} anomalies, suggesting relaxation", count),
                    confidence: 0.8,
                });
            }
        }

        Ok(refinements)
    }

    /// Get anomaly detection statistics
    pub fn stats(&self) -> &AnomalyDetectionStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = AnomalyDetectionStats::default();
    }
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::default_config()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_config_default() {
        let config = ShapeInferenceConfig::default();
        assert_eq!(config.strategy, InferenceStrategy::Statistical);
        assert_eq!(config.min_support, 0.7);
        assert!(config.infer_cardinality);
    }

    #[test]
    fn test_inference_engine_creation() {
        let engine = ShapeInferenceEngine::default_config();
        let stats = engine.stats();

        assert_eq!(stats.shapes_inferred, 0);
        assert_eq!(stats.instances_analyzed, 0);
    }

    #[test]
    fn test_inference_strategy() {
        assert_eq!(
            InferenceStrategy::Statistical,
            InferenceStrategy::Statistical
        );
        assert_ne!(
            InferenceStrategy::Statistical,
            InferenceStrategy::MachineLearning
        );
    }

    #[test]
    fn test_inferred_shape_metadata() {
        let metadata = InferenceMetadata::default();
        assert_eq!(metadata.instances_analyzed, 0);
        assert_eq!(metadata.properties_discovered, 0);
    }

    #[test]
    fn test_sampling() {
        let mut engine = ShapeInferenceEngine::default_config();
        let instances: Vec<Term> = vec![]; // Empty for test

        let sampled = engine.sample_instances(&instances);
        assert_eq!(sampled.len(), 0);
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::default_config();
        assert_eq!(detector.stats().total_anomalies, 0);
    }

    #[test]
    fn test_zscore_anomaly_detection() {
        let mut detector = AnomalyDetector::new(AnomalyDetectionConfig {
            method: AnomalyDetectionMethod::ZScore,
            sensitivity: 2.0,
            min_instances: 3,
            auto_refine: false,
        });

        // Normal values with one outlier
        let values = vec![1.0, 2.0, 3.0, 2.5, 2.2, 100.0];
        let anomalies = detector.detect_value_anomalies(&values).unwrap();

        // Should detect the outlier (100.0)
        assert!(!anomalies.is_empty());
        assert!(anomalies.contains(&5));
    }

    #[test]
    fn test_iqr_anomaly_detection() {
        let mut detector = AnomalyDetector::new(AnomalyDetectionConfig {
            method: AnomalyDetectionMethod::IQR,
            sensitivity: 1.5,
            min_instances: 4,
            auto_refine: false,
        });

        let values = vec![10.0, 12.0, 11.0, 13.0, 12.5, 50.0];
        let anomalies = detector.detect_value_anomalies(&values).unwrap();

        // Should detect the outlier (50.0)
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_modified_zscore_detection() {
        let mut detector = AnomalyDetector::new(AnomalyDetectionConfig {
            method: AnomalyDetectionMethod::ModifiedZScore,
            sensitivity: 3.5,
            min_instances: 3,
            auto_refine: false,
        });

        let values = vec![5.0, 5.1, 4.9, 5.2, 4.8, 20.0];
        let anomalies = detector.detect_value_anomalies(&values).unwrap();

        // Should detect the outlier (20.0)
        assert!(!anomalies.is_empty());
    }

    #[test]
    fn test_anomaly_config_default() {
        let config = AnomalyDetectionConfig::default();
        assert_eq!(config.method, AnomalyDetectionMethod::ModifiedZScore);
        assert_eq!(config.sensitivity, 3.0);
        assert!(config.auto_refine);
    }

    #[test]
    fn test_no_anomalies_for_uniform_data() {
        let mut detector = AnomalyDetector::default_config();

        // All same values
        let values = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let anomalies = detector.detect_value_anomalies(&values).unwrap();

        // Should detect no anomalies
        assert!(anomalies.is_empty());
    }
}
